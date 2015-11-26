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
 * Each work-item invocation of this kernel, calculates the position for 
 * one particle
 *
 */

#define UNROLL_FACTOR  8
#define flt_fix float
#define fix_in int
#define nunsignednt unsigned int 
__kernel 
void nbody_sim(__global float4* pos, __global float4* vel
		,int numBodies ,float deltaTime, float epsSqr
		,__global float4* newPosition, __global float4* newVelocity) {

	unsigned int gid;
	float4 myPos;
	char4 acc; 
	char4 p1;
	char4 r1;
	char invDist1;
	char distSqr1;
	char invDistCube1;
	char s1;
	char4 p2;
	char4 r2;
	char distSqr2;
	char invDist2;
	char invDistCube2;
	char s2;
	char4 oldVel;
	float4 newPosl;
	char4 newVell;

	gid = get_global_id(0);

	myPos = convert_float4(pos[gid]);
	acc = (char4)0.0f;


	fix_in i = 0;
	for (; (i+UNROLL_FACTOR) < numBodies; ) {
	#pragma unroll UNROLL_FACTOR
        	for(int j = 0; j < UNROLL_FACTOR; j++,i++) {
			p1 = convert_char4(pos[i]);
			r1.xyz = convert_char3(p1.xyz) - convert_char3(myPos.xyz);
			distSqr1 = (char)(r1.x * r1.x  +  r1.y * r1.y  +  r1.z * r1.z);
			invDist1 = (char)((1.0f) / sqrt((flt_fix)(distSqr1 + epsSqr)));
			invDistCube1 = (char)(invDist1 * invDist1 * invDist1);
			s1 = (char)(p1.w) * (char)(invDistCube1);

			// accumulate effect of all particles
			acc.xyz += convert_char3(s1 * r1.xyz);
        	}
    	}
	for (; i < numBodies; i++) {
		p2 = convert_char4(pos[i]);

		r2.xyz = convert_char3(p2.xyz) - convert_char3(myPos.xyz);
		distSqr2 = (char)(r2.x * r2.x  +  r2.y * r2.y  +  r2.z * r2.z);

		invDist2 = (char)((1.0f) / sqrt((flt_fix)(distSqr2 + epsSqr)));
		invDistCube2 = (char)(invDist2 * invDist2 * invDist2);
		s2 = (char)(p2.w) * (char)(invDistCube2);

		// accumulate effect of all particles
		acc.xyz += convert_char3(s2 * r2.xyz);
	}

	oldVel = convert_char4(vel[gid]);

	// updated position and velocity
	newPosl.xyz = convert_float3(myPos.xyz) + convert_float3(oldVel.xyz * deltaTime) + convert_float3(acc.xyz * 0.5f * deltaTime * deltaTime);
	newPosl.w = (float)(myPos.w);

	newVell.xyz = convert_char3(oldVel.xyz) + convert_char3(acc.xyz * deltaTime);
	newVell.w = (char)(oldVel.w);

	// write to global memory
	newPosition[gid] = convert_float4(newPosl);
	newVelocity[gid] = convert_float4(newVell);
}
