// const float ics = 1.0f/solid.cs;
// const float ich = 1.0f/solid.ch;
// // Rasterize triangles.
// for (int i = 0; i < nt; ++i)
// {
//     const float* v0 = &verts[tris[i*3+0]*3];
//     const float* v1 = &verts[tris[i*3+1]*3];
//     const float* v2 = &verts[tris[i*3+2]*3];
//     // Rasterize.
//     if (!rasterizeTri(v0, v1, v2, areas[i], solid, solid.bmin, solid.bmax, solid.cs, ics, ich, flagMergeThr))
//     {
//         ctx->log(RC_LOG_ERROR, "rcRasterizeTriangles: Out of memory.");
//         return false;
//     }
// }

// static bool rasterizeTri(const float* v0, const float* v1, const float* v2,
// 						 const unsigned char area, rcHeightfield& hf,
// 						 const float* bmin, const float* bmax,
// 						 const float cs, const float ics, const float ich,
// 						 const int flagMergeThr)
// {
// 	const int w = hf.width;
// 	const int h = hf.height;
// 	float tmin[3], tmax[3];
// 	const float by = bmax[1] - bmin[1];
// 	
// 	// Calculate the bounding box of the triangle.
// 	rcVcopy(tmin, v0);
// 	rcVcopy(tmax, v0);
// 	rcVmin(tmin, v1);
// 	rcVmin(tmin, v2);
// 	rcVmax(tmax, v1);
// 	rcVmax(tmax, v2);
// 	
// 	// If the triangle does not touch the bbox of the heightfield, skip the triagle.
// 	if (!overlapBounds(bmin, bmax, tmin, tmax))
// 		return true;
// 	
// 	// Calculate the footprint of the triangle on the grid's y-axis
// 	int y0 = (int)((tmin[2] - bmin[2])*ics);
// 	int y1 = (int)((tmax[2] - bmin[2])*ics);
// 	y0 = rcClamp(y0, 0, h-1);
// 	y1 = rcClamp(y1, 0, h-1);
// 	
// 	// Clip the triangle into all grid cells it touches.
// 	float buf[7*3*4];
// 	float *in = buf, *inrow = buf+7*3, *p1 = inrow+7*3, *p2 = p1+7*3;
// 
// 	rcVcopy(&in[0], v0);
// 	rcVcopy(&in[1*3], v1);
// 	rcVcopy(&in[2*3], v2);
// 	int nvrow, nvIn = 3;
// 	
// 	for (int y = y0; y <= y1; ++y)
// 	{
// 		// Clip polygon to row. Store the remaining polygon as well
// 		const float cz = bmin[2] + y*cs;
// 		dividePoly(in, nvIn, inrow, &nvrow, p1, &nvIn, cz+cs, 2);
// 		rcSwap(in, p1);
// 		if (nvrow < 3) continue;
// 		
// 		// find the horizontal bounds in the row
// 		float minX = inrow[0], maxX = inrow[0];
// 		for (int i=1; i<nvrow; ++i)
// 		{
// 			if (minX > inrow[i*3])	minX = inrow[i*3];
// 			if (maxX < inrow[i*3])	maxX = inrow[i*3];
// 		}
// 		int x0 = (int)((minX - bmin[0])*ics);
// 		int x1 = (int)((maxX - bmin[0])*ics);
// 		x0 = rcClamp(x0, 0, w-1);
// 		x1 = rcClamp(x1, 0, w-1);
// 
// 		int nv, nv2 = nvrow;
// 
// 		for (int x = x0; x <= x1; ++x)
// 		{
// 			// Clip polygon to column. store the remaining polygon as well
// 			const float cx = bmin[0] + x*cs;
// 			dividePoly(inrow, nv2, p1, &nv, p2, &nv2, cx+cs, 0);
// 			rcSwap(inrow, p2);
// 			if (nv < 3) continue;
// 			
// 			// Calculate min and max of the span.
// 			float smin = p1[1], smax = p1[1];
// 			for (int i = 1; i < nv; ++i)
// 			{
// 				smin = rcMin(smin, p1[i*3+1]);
// 				smax = rcMax(smax, p1[i*3+1]);
// 			}
// 			smin -= bmin[1];
// 			smax -= bmin[1];
// 			// Skip the span if it is outside the heightfield bbox
// 			if (smax < 0.0f) continue;
// 			if (smin > by) continue;
// 			// Clamp the span to the heightfield bbox.
// 			if (smin < 0.0f) smin = 0;
// 			if (smax > by) smax = by;
// 			
// 			// Snap the span to the heightfield height grid.
// 			unsigned short ismin = (unsigned short)rcClamp((int)floorf(smin * ich), 0, RC_SPAN_MAX_HEIGHT);
// 			unsigned short ismax = (unsigned short)rcClamp((int)ceilf(smax * ich), (int)ismin+1, RC_SPAN_MAX_HEIGHT);
// 			
// 			if (!addSpan(hf, x, y, ismin, ismax, area, flagMergeThr))
// 				return false;
// 		}
// 	}
// 
// 	return true;
// }
// 

// inline bool overlapBounds(const float* amin, const float* amax, const float* bmin, const float* bmax)
// {
// 	bool overlap = true;
// 	overlap = (amin[0] > bmax[0] || amax[0] < bmin[0]) ? false : overlap;
// 	overlap = (amin[1] > bmax[1] || amax[1] < bmin[1]) ? false : overlap;
// 	overlap = (amin[2] > bmax[2] || amax[2] < bmin[2]) ? false : overlap;
// 	return overlap;
// }

// todo: To prevent register spilling, stuff of the same type can be put into a single buffer.
__kernel void rasterize_tris(__global const float* verts, 
						  	 __global const int* tris,
						  	 __global const unsigned char* areas,
						     const float3 hf_bmin,
						     const float3 hf_bmax,
						     const float hf_cs,
						     const float hf_ch,
						     const int hf_width,
						     const int hf_height,
						     const int flagMergeThr)
{
	if(get_global_id(0) == 0)
	{
		printf("Heightfield:\n  min: %f, %f, %f\n  max: %f, %f, %f\n  cs: %f\n  ch: %f\n, width: %d\n  height: %d\n",
			hf_bmin[0], hf_bmin[1], hf_bmin[2], hf_bmax[0], hf_bmax[1], hf_bmax[2], hf_cs, hf_ch, hf_width, hf_height);
	}

	// Extract triangle vertices from input.
	float3 v0, v1, v2;
	{
		int tidx = get_global_id(0);
		// Triangles are (v0index, v1index, v2index), packed as a flat array of integers.
		// todo: Use vector load functions here
		int mtidx = tidx * 3; 
		int vidx0 = tris[mtidx] * 3;
		int vidx1 = tris[mtidx + 1] * 3;
		int vidx2 = tris[mtidx + 2] * 3;
		v0 = (float3)(verts[vidx0], verts[vidx0 + 1], verts[vidx0 + 2]);
		v1 = (float3)(verts[vidx1], verts[vidx1 + 1], verts[vidx1 + 2]);
		v2 = (float3)(verts[vidx2], verts[vidx2 + 1], verts[vidx2 + 2]);
		printf("Processing triangle %d (vert idcs %d, %d, %d):"
			"\n  %v3f\n  %v3f\n  %v3f\n", 
			tidx, vidx0, vidx1, vidx2,
			v0, v1, v2);
	}

	// Calculate the bounding box of the triangle
	float3 tmin = fmin(fmin(v0, v1), v2);
	float3 tmax = fmax(fmax(v0, v1), v2);
	printf("Bounding box min: %v3f\n", tmin);
	printf("Bounding box max: %v3f\n", tmax);

	// If the triangle doesn't overlap with the bounds of heightfield, discard it.
	// todo: This may be worth doing as a preprocessing step, to remove divergence.
	{
		int3 cmp = hf_bmin > tmax || hf_bmax < tmin;
		printf("Overlap \n  [%.2v3f][%.2v3f] vs [%.2v3f][%.2v3f]\n  %v3d\n", hf_bmin, hf_bmax, tmin, tmax, cmp);
		if(!all(cmp))
		{
			return;
		}
	}
}