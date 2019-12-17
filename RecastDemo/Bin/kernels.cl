void dividePoly(const float3* in, int nin,
				float3* out1, int* nout1,
				float3* out2, int* nout2,
				float x, int axis)
{
	float d[12];
	for (int i = 0; i < nin; ++i)
	{	
		float unroll[] = {in[i].x, in[i].y, in[i].z};
		d[i] = x - unroll[axis];
	}

	int m = 0, n = 0;
	for (int i = 0, j = nin-1; i < nin; j=i, ++i)
	{
		bool ina = d[j] >= 0;
		bool inb = d[i] >= 0;
		if (ina != inb)
		{
			float s = d[j] / (d[j] - d[i]);
			out1[m] = in[j] + (in[i] - in[j])*s;
			*(out2 + n) = *(out1 + m);
			m++;
			n++;

			// add the i'th point to the right polygon. Do NOT add points that are on the dividing line
			// since these were already added above
			if (d[i] > 0)
			{
				*(out1 + m) = *(in + i);
				m++;
			}
			else if (d[i] < 0)
			{
				*(out2 + n) = *(in + i);
				n++;
			}
		}
		else // same side
		{
			// add the i'th point to the right polygon. Addition is done even for points on the dividing line
			if (d[i] >= 0)
			{
				*(out1 + m) = *(in + i);
				m++;
				if (d[i] != 0)
					continue;
			}
			*(out2 + n) = *(in + i);
			n++;
		}
	}

	*nout1 = m;
	*nout2 = n;		
}

void swap_ptrs(float3** a, float3** b)
{
	float3* tmp = *a;
	*a = *b;
	*b = tmp;
}

void serialize_int(int val, __global unsigned char* buf, int* inOut_oidx)
{
	// todo[drabard]: endianness
	int oidx = *inOut_oidx;
	int ii = 0;
	for(int oi = 0; oi < 4; ++oi)
	{
		buf[oidx + oi] = ((unsigned char*)(&val))[ii];
		++ii;
	}

	(*inOut_oidx) += 4;
}

void serialize_ushort(ushort val, __global unsigned char* buf, int* inOut_oidx)
{
	// todo[drabard]: endianness
	int oidx = *inOut_oidx;
	int ii = 0;
	for(int oi = 0; oi < 2; ++oi)
	{
		buf[oidx + oi] = ((unsigned char*)(&val))[ii];
		++ii;
	}

	(*inOut_oidx) += 2;
}

void add_output(int x, int y, 
				ushort ismin, ushort ismax, 
				int tidx, int max_spans_per_tri, 
				int addedSpans, 
				__global unsigned char* out)
{
	// total bytes to write: 
	// 4 tidx
	// 4 x
	// 4 y
	// 2 ismin
	// 2 ismax
	// 16 bytes total
	int entry_size = 16;
	int tri_offset = tidx*max_spans_per_tri * entry_size;
	int oidx = tri_offset + addedSpans * entry_size;
	serialize_int(tidx, out, &oidx);
	serialize_int(x, out, &oidx);
	serialize_int(y, out, &oidx);
	serialize_ushort(ismin, out, &oidx);
	serialize_ushort(ismax, out, &oidx);
}

// todo: To prevent register spilling, stuff of the same type can be put into a single buffer.
__kernel void rasterize_tris(__global const float* verts, 
						  	 __global const int* tris,
						  	 __global unsigned char* out,
						     const float3 hf_bmin,
						     const float3 hf_bmax,
						     const float hf_cs,
						     const float hf_ch,
						     const int nt,
						     const int hf_width, // extent of the heightfield in x dimension
						     const int hf_height, // extent of the heightfield in z dimension
						     const int max_span_height,
						     const int max_spans_per_tri)
{
	int tidx = get_global_id(0);
	if(tidx >= nt) return;

	// Extract triangle vertices from input.
	// todo: Use vector load functions here
	float3 v0, v1, v2;
	{
		int tidx = get_global_id(0);
		// Triangles are (v0index, v1index, v2index), packed as a flat array of integers.
		int mtidx = tidx * 3; 
		int vidx0 = tris[mtidx] * 3;
		int vidx1 = tris[mtidx + 1] * 3;
		int vidx2 = tris[mtidx + 2] * 3;
		v0 = (float3)(verts[vidx0], verts[vidx0 + 1], verts[vidx0 + 2]);
		v1 = (float3)(verts[vidx1], verts[vidx1 + 1], verts[vidx1 + 2]);
		v2 = (float3)(verts[vidx2], verts[vidx2 + 1], verts[vidx2 + 2]);
	}

	// Calculate the bounding box of the triangle
	float3 tmin = fmin(fmin(v0, v1), v2);
	float3 tmax = fmax(fmax(v0, v1), v2);

	// If the triangle doesn't overlap with the bounds of heightfield, discard it.
	// todo: This may be worth doing as a preprocessing step, to remove divergence.
	{
		int3 cmp = hf_bmin <= tmax && hf_bmax >= tmin;
		
		if(!all(cmp))
		{
			add_output(-1, -1, 0, 0, tidx, max_spans_per_tri, 0, out);
			return;
		}
	}

	// Find bottom and top z boundaries.
	float ics = 1.0f/hf_cs;
	float ich = 1.0f/hf_ch;
	int y0 = (int)((tmin.z - hf_bmin.z) * ics);
	int y1 = (int)((tmax.z - hf_bmin.z) * ics);
	y0 = clamp(y0, 0, hf_height-1);
	y1 = clamp(y1, 0, hf_height-1);
	
	// Clip the triangle into all grid cells it touches.
	// [- - - - - - -   in
	//	- - - - - - -	inrow
	//	- - - - - - -   p1
	//  - - - - - - -]  p2
	float3 buf[7*4];
	float3* in = buf;
	float3* inrow = buf+7;
	float3* p1 = inrow+7;
	float3* p2 = p1+7;

	in[0] = v0;
	in[1] = v1;
	in[2] = v2;
	int nvrow, nvIn = 3;

	float by = hf_bmax.y - hf_bmin.y;
	
	int addedSpans = 0;
	for (int y = y0; y <= y1; ++y)
	{
		// Clip polygon to row. Store the remaining polygon as well
		const float cy = hf_bmin.z + y*hf_cs;

		int pnvIn = nvIn;
		dividePoly(in, nvIn, inrow, &nvrow, p1, &nvIn, cy+hf_cs, 2);

		swap_ptrs(&in, &p1);
		if (nvrow < 3) continue;

		float minX = inrow[0].x;
		float maxX = inrow[0].x;

		// find the horizontal bounds in the row
		for (int i=1; i<nvrow; ++i)
		{
			minX = min(minX, inrow[i].x);
			maxX = max(maxX, inrow[i].x);
		}

		int x0 = (int)((minX - hf_bmin.x)*ics);
		int x1 = (int)((maxX - hf_bmin.x)*ics);
		x0 = clamp(x0, 0, hf_width-1);
		x1 = clamp(x1, 0, hf_width-1);
		int nv, nv2 = nvrow;

		for (int x = x0; x <= x1; ++x)
		{
			// Clip polygon to column. store the remaining polygon as well
			const float cx = hf_bmin.x + x*hf_cs;
			int pnv2 = nv2;
			dividePoly(inrow, nv2, p1, &nv, p2, &nv2, cx+hf_cs, 0);

			swap_ptrs(&inrow, &p2);
			if (nv < 3) continue;

			// Calculate min and max of the span.
			float smin = p1[0].y;
			float smax = p1[0].y;
			for (int i = 1; i < nv; ++i)
			{
				smin = min(smin, p1[i].y);
				smax = max(smax, p1[i].y);
			}
			smin -= hf_bmin.y;
			smax -= hf_bmin.y;

			// Skip the span if it is outside the heightfield bbox
			if (smax < 0.0f) continue;
			if (smin > by) continue;

			// Clamp the span to the heightfield bbox.
			smin = max(smin, 0.0f);
			smax = min(smax, by);

			// Snap the span to the heightfield height grid.
			ushort ismin = (ushort)clamp((int)floor(smin * ich), 0, max_span_height);
			ushort ismax = (ushort)clamp((int)ceil(smax * ich), (int)ismin+1, max_span_height);	

			if(addedSpans < max_spans_per_tri)
			{
				add_output(x, y, ismin, ismax, tidx, max_spans_per_tri, addedSpans, out);
				++addedSpans;
			}
		}
	}

	if(addedSpans < max_spans_per_tri)
	{
		add_output(-1, -1, 0, 0, tidx, max_spans_per_tri, addedSpans, out);
	}
}