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

/**
 * SimpleConvolution is where each pixel of the output image
 * is the weighted sum of the neighborhood pixels of the input image
 * The neighborhood is defined by the dimensions of the mask and 
 * weight of each neighbor is defined by the mask itself.
 * @param output Output matrix after performing convolution
 * @param input  Input  matrix on which convolution is to be performed
 * @param mask   mask matrix using which convolution was to be performed
 * @param inputDimensions dimensions of the input matrix
 * @param maskDimensions  dimensions of the mask matrix
 */

#define ncuin uint
__kernel void simpleConvolution(__global  uint  * output,
                                __global  uint  * input,
                                __global  float  * mask,
                                const     uint2  inputDimensions,
                                const     uint2  maskDimensions)
{	
	ncuin tid;
	uchar width;
	uchar height;
	uchar varx;
	uchar vary;
	uchar maskWidth; 
	uchar maskHeight;
	uchar vstep;
	uchar hstep;
	uchar left;
	uchar right;
	uchar top;
	uchar bottom; 
	float sumFX;
	uchar maskIndex;
	ushort index; 
	uchar iindex;
	uchar jindex;

	tid   = get_global_id(0);
    
	width  = inputDimensions.x;
	height = inputDimensions.y;
    
	varx      = tid%width;
	vary      = tid/width;
    
	maskWidth  = maskDimensions.x;
	maskHeight = maskDimensions.y;
    
	vstep = (maskWidth  -1)/2;
	hstep = (maskHeight -1)/2;
    
    /*
     * find the left, right, top and bottom indices such that
     * the indices do not go beyond image boundaires
     */
	left    = (varx           <  vstep) ? 0         : (varx - vstep);
	right   = ((varx + vstep) >= width) ? width - 1 : (varx + vstep); 
	top     = (vary           <  hstep) ? 0         : (vary - hstep);
	bottom  = ((vary + hstep) >= height)? height - 1: (vary + hstep); 
    
    /*
     * initializing wighted sum value
     */
	sumFX = 0;
  
	for(iindex = left; iindex <= right; ++iindex)
		for(uint jindex = top ; jindex <= bottom; ++jindex)    
        	{
            /*
             * performing wighted sum within the mask boundaries
             */
            		maskIndex = (jindex - (vary - hstep)) * maskWidth  + (iindex - (varx - vstep));
            		index     = jindex                 * width      + iindex;
            		sumFX += (input[index] * mask[maskIndex]);
        	}
    /* 
     *To round to the nearest integer
     */
	output[tid] = (uint)(sumFX) + 1;
}
