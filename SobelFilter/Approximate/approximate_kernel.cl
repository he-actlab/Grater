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

__kernel void sobel_filter(__global uchar4* restrict inputImage, __global uchar4* restrict outputImage, const uint one, const uint two )
{
	uint x = get_global_id(0);
    uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);

	short4 Gx;
	short4 Gy;
	int4 i00;
	short4 i10;
	short4 i20;
	short4 i01;
	short4 i11;
	short4 i21;
	short4 i02;
	short4 i12;
	short4 i22;

	Gx = (short4)(0);
	Gy = convert_short4(Gx);
	
	int c = x + y * width;


	/* Read each texel component and calculate the filtered value using neighbouring texel components */
	if( x >= 1 && x < (width-1) && y >= 1 && y < height - 1)
	{
		i00 = convert_int4(inputImage[c - 1 - width]);
		i10 = convert_short4(inputImage[c - width]);
		i20 = convert_short4(inputImage[c + 1 - width]);
		i01 = convert_short4(inputImage[c - 1]);
		i11 = convert_short4(inputImage[c]);
		i21 = convert_short4(inputImage[c + 1]);
		i02 = convert_short4(inputImage[c - 1 + width]);
		i12 = convert_short4(inputImage[c + width]);
		i22 = convert_short4(inputImage[c + 1 + width]);

		Gx =  (short4)(one) * convert_short4(i00) + (short4)(two) * convert_short4(i10) + (short4)(one) * convert_short4(i20) - (short4)(one) * convert_short4(i02)  - (short4)(two) * convert_short4(i12) -(short4)(one) *  convert_short4(i22);

		Gy =  (short4)(one) *  convert_short4(i00) - (short4)(one) * convert_short4(i20)  + (short4)(two)*convert_short4(i01) - (short4)(two)*convert_short4(i21) +(short4)(one) *  convert_short4(i02)  -  (short4)(one) * convert_short4(i22);


		
		outputImage[c] = convert_uchar4(hypot(convert_float4(Gx), convert_float4(Gy))/(float4)(2));


	}
			
}

	

	 






	

	




	

	

	
	
