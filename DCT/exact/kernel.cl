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
 * Perform Discrete Cosine Transform for block of size 8x8
 * in the input matrix
 * @param output output of the DCT8x8 transform 
 * @param input  input array 
 * @param dct8x8 8x8 consine function base used to calculate DCT8x8
 * @param inper  local memory which stores intermediate result
 * @param width  width of the input matrix
 * @param blockWidth width of each block, 8 here
 * @param inverse  flag to perform inverse DCT
 */

__kernel 
void DCT(__global float * output,
         __global float * input, 
         __global float * dct8x8,
         __local  float * inper,
         const    uint    width,
         const    uint  blockWidth,
         const    uint    inverse)

{
    /* get global indices of the element */
	uint globalIdx; 
	uint globalIdy;
	uint groupIdx;
	uint groupIdy;
	uint iindex;
	uint jindex; 
	uint idx;
	float acc;
	uint index1;
	uint getIndxGlobalIdx;
	uint getIndxGlobalIdy; 
	uint index2; 
	uint findex1;
	uint findex2;
 
	globalIdx = get_global_id(0);
	globalIdy = get_global_id(1);

    /* get indices of the block to which the element belongs to */
	groupIdx  = get_group_id(0);
	groupIdy  = get_group_id(1);

    /* get indices relative to the block */
	iindex  = get_local_id(0);
	jindex  = get_local_id(1);
    
	idx = globalIdy * width + globalIdx;

    /* initialise the accumulator */
    acc = 0.0f;
    
    /* AT * X  */
    for(uint k=0; k < blockWidth; k++)
    {
	index1 = (inverse)? iindex*blockWidth + k : k * blockWidth + iindex;
	getIndxGlobalIdx = groupIdx * blockWidth + jindex;
	getIndxGlobalIdy = groupIdy * blockWidth + k;
	index2 = (getIndxGlobalIdy * width  + getIndxGlobalIdx);

        //uint index2 = getIdx(groupIdx, groupIdy, jindex, k, blockWidth, width);
        
        acc += dct8x8[index1] * input[index2];
    }
    inper[jindex*blockWidth + iindex] = acc;

    /* 
     * Make sure all the values of inter that belong to a block 
     * are calculated before proceeding further 
     */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* again initalising the accumulator */
    acc = 0.0f;
    
    /* (AT * X) * A */
    for(uint k=0; k < blockWidth; k++)
    {
	findex1 = iindex* blockWidth + k; 
	findex2 = (inverse)? jindex*blockWidth + k : k* blockWidth + jindex;
        
        acc += inper[findex1] * dct8x8[findex2];
    }
    output[idx] = acc;    
}
