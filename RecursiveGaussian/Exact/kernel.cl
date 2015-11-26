/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

/*
 * Transpose Kernel 
 * input image is transposed by reading the data into a block
 * and writing it to output image
 */

#define ncuin uint

__kernel 
void transpose_kernel(__global uchar4 *output,
                      __global uchar4  *input,
                      __local  uchar4 *block,
                      const    uint    width,
                      const    uint    height,
                      const    uint blockSize)
{
	ncuin globalIdx = get_global_id(0);
	ncuin globalIdy = get_global_id(1);
	
	ncuin localIdx = get_local_id(0);
	ncuin localIdy = get_local_id(1);
	
    /* copy from input to local memory */
	block[localIdy * blockSize + localIdx] = input[globalIdy*width + globalIdx];

    /* wait until the whole block is filled */
	barrier(CLK_LOCAL_MEM_FENCE);

    /* calculate the corresponding raster indices of source and target */
	ncuin sourceIndex = localIdy * blockSize + localIdx;
	ncuin targetIndex = globalIdy + globalIdx * height; 
	
	output[targetIndex] = block[sourceIndex];
}




/*  Recursive Gaussian filter
 *  parameters:	
 *      input - pointer to input data 
 *      output - pointer to output data 
 *      width  - image width
 *      iheight  - image height
 *      a0-a3, b1, b2, coefp, coefn - gaussian parameters
 */
__kernel void RecursiveGaussian_kernel(__global const uchar4* input, __global uchar4* output, 
				       const int width, const int height, 
				       const float a0, const float a1, 
				       const float a2, const float a3, 
				       const float b1, const float b2, 
				       const float coefp, const float coefn)
{
    // compute x : current column ( kernel executes on 1 column )
	unsigned int x = get_global_id(0);

    if (x >= width) 
	return;
	
    // start forward filter pass

	float4 xp;
	float4 yp;
	float4 yb;
	float4 xc;
	float4 yc;
	float4 xn;
	float4 xa;
	float4 yn;
	float4 ya;
	float4 cx;
	float4 cy;
	float4 temp;
	int pos;

	xp = (float4)0.0f;  // previous input
	yp = (float4)0.0f;  // previous output
	yb = (float4)0.0f;  // previous output by 2

    for (int y = 0; y < height; y++) 
    {
	  pos = (int)(x + y * width);
          xc = (float4)(input[pos].x, input[pos].y, input[pos].z, input[pos].w);
          yc = convert_float4(convert_float4((a0 * convert_float4(xc))) + convert_float4((a1 * convert_float4(xp))) - convert_float4((b1 * convert_float4(yp))) - convert_float4((b2 * convert_float4(yb))));
	  output[pos] = (uchar4)(yc.x, yc.y, yc.z, yc.w);
	  xp = convert_float4(xc); 
          yb = convert_float4(yp); 
          yp = convert_float4(yc); 
    }

     barrier(CLK_GLOBAL_MEM_FENCE);


    // start reverse filter pass: ensures response is symmetrical
	xn = (float4)(0.0f);
	xa = (float4)(0.0f);
	yn = (float4)(0.0f);
	ya = (float4)(0.0f);


    for (int y = height - 1; y > -1; y--) 
    {
        pos = (int)(x + y * width);
	cx =  (float4)(input[pos].x, input[pos].y, input[pos].z, input[pos].w);
	cy = convert_float4(convert_float4((a2 * convert_float4(xn))) + convert_float4((a3 * convert_float4(xa))) - convert_float4((b1 * convert_float4(yn))) - convert_float4((b2 * convert_float4(ya))));
        xa = convert_float4(xn); 
        xn = convert_float4(cx); 
        ya = convert_float4(yn); 
        yn = convert_float4(cy);
	  temp = (float4)(output[pos].x, output[pos].y, output[pos].z, output[pos].w) + convert_float4(cy);
	  output[pos] = (uchar4)(temp.x, temp.y, temp.z, temp.w);

    }
}






	 






	

	




	

	

	
	
