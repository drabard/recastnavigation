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

void add_output(int x, int y, 
				ushort ismin, ushort ismax, 
				int tidx, int max_spans_per_tri, 
				int addedSpans, 
				__global int* out_xy, __global ushort* out_sminmax)
{
	int tri_offset = tidx*max_spans_per_tri;
	int oidx2 = tri_offset + addedSpans * 2;
	out_xy[oidx2] = x;
	out_xy[oidx2 + 1] = y;
	out_sminmax[oidx2] = ismin;
	out_sminmax[oidx2 + 1] = ismax;
}

// todo: To prevent register spilling, stuff of the same type can be put into a single buffer.
__kernel void rasterize_tris(__global const float* verts, 
						  	 __global const int* tris,
						  	 __global int* out_xy,
						  	 __global ushort* out_sminmax,
						     const float3 hf_bmin,
						     const float3 hf_bmax,
						     const float hf_cs,
						     const float hf_ch,
						     const int hf_width, // extent of the heightfield in x dimension
						     const int hf_height, // extent of the heightfield in z dimension
						     const int max_span_height,
						     const int max_spans_per_tri)
{
	int tidx = get_global_id(0);
	if(get_global_id(0) == 0)
	{
		printf("[%d] Heightfield:\n  min: %v3f\n  max: %v3f\n  cs: %f\n  ch: %f\n  width: %d\n  height: %d\n  max span height: %d\n",
			tidx, hf_bmin, hf_bmax, hf_cs, hf_ch, hf_width, hf_height, max_span_height);
	}

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
		printf("[%d] ====\nProcessing triangle %d (vert idcs %d, %d, %d):"
			"\n  %v3f\n  %v3f\n  %v3f\n",
			tidx, tidx, vidx0, vidx1, vidx2,
			v0, v1, v2);
	}

	// Calculate the bounding box of the triangle
	float3 tmin = fmin(fmin(v0, v1), v2);
	float3 tmax = fmax(fmax(v0, v1), v2);
	printf("[%d] Bounding box min: %v3f\n", tidx, tmin);
	printf("[%d] Bounding box max: %v3f\n", tidx, tmax);

	// If the triangle doesn't overlap with the bounds of heightfield, discard it.
	// todo: This may be worth doing as a preprocessing step, to remove divergence.
	{
		int3 cmp = hf_bmin <= tmax && hf_bmax >= tmin;
		printf("[%d] Overlap \n  [%.2v3f][%.2v3f] vs [%.2v3f][%.2v3f]\n  %v3d\n", tidx, hf_bmin, hf_bmax, tmin, tmax, cmp);
		if(!all(cmp))
		{
			printf("[%d] No overlap, discarding triangle.\n====\n", tidx);

			// Add terminating span.
			add_output(-1, -1, 0, 0, tidx, max_spans_per_tri, 0, out_xy, out_sminmax);
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

	printf("[%d] y0: %d, y1: %d\n", tidx, y0, y1);

	// if(tidx == 0)
	// {
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
	printf("[%d] by: %f\n", tidx, by);

	int addedSpans = 0;
	for (int y = y0; y <= y1; ++y)
	{
		// Clip polygon to row. Store the remaining polygon as well
		const float cy = hf_bmin.z + y*hf_cs;

		int pnvIn = nvIn;
		dividePoly(in, nvIn, inrow, &nvrow, p1, &nvIn, cy+hf_cs, 2);
		if(tidx == 1)
		{
			printf("dividePoly, x: %f, axis: %d\n  ", cy+hf_cs, 2);
			for(int i = 0; i < pnvIn; ++i)
			{
				printf("[%v3f] ", in[i]);
			}
			printf("\n");
			printf("  inside: ");
			for(int i = 0; i < nvrow; ++i)
			{
				printf("[%v3f] ", inrow[i]);
			}
			printf("\n");
			printf("  outside: ");
			for(int i = 0; i < nvIn; ++i)
			{
				printf("[%v3f] ", p1[i]);
			}
			printf("\n");
		}

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
		swap_ptrs(&inrow, &p2);
		for (int x = x0; x <= x1; ++x)
		{
			// Clip polygon to column. store the remaining polygon as well
			const float cx = hf_bmin.x + x*hf_cs;
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
			printf("[%d] smin, smax: %f, %f\n", tidx, smin, smax);

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
				add_output(x, y, ismin, ismax, tidx, max_spans_per_tri, addedSpans, out_xy, out_sminmax);
				++addedSpans;
			}
		}
	}

	printf("[%d] Added %d spans", tidx, addedSpans);

	// Add terminating span.
	if(addedSpans < max_spans_per_tri)
	{
		add_output(-1, -1, 0, 0, tidx, max_spans_per_tri, addedSpans, out_xy, out_sminmax);
	}
}