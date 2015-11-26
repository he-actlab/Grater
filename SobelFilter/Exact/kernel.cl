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
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each thread calculates a pixel component(rgba), by applying a filter 
 * on group of 8 neighbouring pixels in both x and y directions. 
 * Both filters are summed (vector sum) to form the final result.
 */
#define ncuin uint

__kernel void sobel_filter(__global uchar4* restrict inputImage, __global uchar4* restrict outputImage, const uint one, const uint two )
{
	ncuin x = get_global_id(0);
    	ncuin y = get_global_id(1);

	ncuin width = get_global_size(0);
	ncuin height = get_global_size(1);

	float4 Gx;
	float4 Gy;
	float4 i00;
	float4 i10;
	float4 i20;
	float4 i01;
	float4 i11;
	float4 i21;
	float4 i02;
	float4 i12;
	float4 i22;
	int var_c;

	Gx = (float4)(0);
	Gy = convert_float4(Gx);
	
	var_c = x + y * width;

	if( x >= 1 && x < (width-1) && y >= 1 && y < height - 1)
	{
		i00 = convert_float4(inputImage[var_c - 1 - width]);
		i10 = convert_float4(inputImage[var_c - width]);
		i20 = convert_float4(inputImage[var_c + 1 - width]);
		i01 = convert_float4(inputImage[var_c - 1]);
		i11 = convert_float4(inputImage[var_c]);
		i21 = convert_float4(inputImage[var_c + 1]);
		i02 = convert_float4(inputImage[var_c - 1 + width]);
		i12 = convert_float4(inputImage[var_c + width]);
		i22 = convert_float4(inputImage[var_c + 1 + width]);

		Gx =  convert_float4(one * i00) + convert_float4(two * i10) + convert_float4(one * i20) - convert_float4(one * i02)  - convert_float4(two * i12) - convert_float4(one * i22);

		Gy =  convert_float4(one * i00) - convert_float4(one * i20)  + convert_float4(two * i01) - convert_float4(two * i21) + convert_float4(one * i02)  - convert_float4(one * i22);
		
		outputImage[var_c] = convert_uchar4(hypot(convert_float4(Gx), convert_float4(Gy))/2);

	}
			
}
