from operator import itemgetter, attrgetter
import random
import sys
import os
import math
import re

# GLOBAL VARIABLES

solution_found = False
popN = 50#50 # n number of chromos per population
genesPerCh = 75
max_iterations = 30#50
target = 1111.0
crossover_rate = 0.7
mutation_rate = 0.05
ELITISM_RATE = 15 #15
NumberOfTrainingSets = 100

values4 = ['char4','short4','int4','float4']
values3 = ['char3', 'short3', 'int3', 'float3']
values = ['char', 'short', 'int', 'float']
u_values = ['uchar', 'ushort', 'uint']

global numFloats
global numInts
global numShorts
global numChars
global bestFound

numFloats, numInts, numShorts, numChars = [], [], [], []
bestFound = []

"""Generates random population of chromos"""
def generatePop (numVars, bestFound, variableList, variableType, inputFileName):
#	bestFound = [3 for i in xrange(numVars)]
#	bestFound = []
	for i in range(numVars):
		bestFound.append(exactBound[i])
#	print "bestFound = ", bestFound
	chromos, chromo = [], []
#	print numVars
	numChromo = 0
	#for eachChromo in range(popN):
	while numChromo < popN:
		chromo = []
		for bit in range(numVars):#(genesPerCh * 4):
			if (variableType[bit] == 'float4' or variableType[bit] == 'float' or variableType[bit] =='float3'):	
				chromo.append(random.randint(lowerBounds[bit],3))
#				print "variableType = " , variableType[bit]
			elif (variableType[bit] == 'int4' or variableType[bit] == 'int' or variableType[bit] == 'int3' or variableType[bit] == 'uint'): # or variableType[bit] == 'uint'):
				chromo.append(random.randint(lowerBounds[bit],2))
			elif (variableType[bit] == 'short4' or variableType[bit] == 'short' or variableType[bit] == 'short3' or variableType[bit] == 'ushort'): # or variableType[bit] == 'ushort'):
				chromo.append(random.randint(lowerBounds[bit],1))

#		print "chromo = ", chromo
# run the kernel to see if it is passed or not --> I want to keep only the kernels that pass
		isAcc = accuracyCheck(chromo, bestFound, numVars) # isAcc = -1 : can't decide w/o execution, = 0: can decide/not accurate, = 1: can decide/accurate
	#	print "isAcc = ", isAcc
		if (isAcc == -1):		 
			passed = executeKernel(chromo, numChromo, numVars, variableList, variableType, inputFileName)
		elif (isAcc == 1):
			passed = 1
	#		print "WOW: no execution, passed!"
		elif (isAcc == 0):
			passed = 0
	#		print "WOW: no execution, failed!"
		if (passed == 1): 
			#print numChromo, chromo
			chromos.append(chromo)
			numChromo = numChromo + 1
			bestFound = updateBestFound(bestFound, chromo) 
#	print chromos
	return chromos

def updateBestFound(bestFound, chromo):
	isLess = 1
	for i in range (0, len(chromo)):
		if (chromo[i] > bestFound[i]): 
			isLess = 0
	if (isLess == 1):
#		print "previous bestFound = " , bestFound
#		print "chrom is = ", chromo
#		bestFound = chromo
		for i in range (0, len(bestFound)):
			bestFound[i] = chromo[i]
#		print "bestFound is = ", bestFound
	return bestFound


def accuracyCheck(chromo, bestFound, numVars):
	isCond1 = 1
	for i in range(numVars):
		if (chromo[i] < bestFound[i]):
			isCond1 = 0
	isCond2 = 0
	for i in range(numVars):
		if (chromo[i] < lowerBounds[i]):
			isCond2 = 1
		if (bestFound[i] < lowerBounds[i]):
			print "Oh my!! something is wrong here! bestFound is lower than lowerBounds"
	isAcc = -1
	if (isCond1 == 1):
		isAcc = 1
	elif (isCond2 == 1):
		isAcc = 0
	return isAcc

"""Read the PSNR from out file"""
def readPSNR(numChromo):	
	filename = "out" + str(numChromo) + ".log"
	outfile = open (filename,'r')
	data2 = outfile.readlines()
	psnr = 0
	for line in data2:	
		if (line.strip().startswith('PSNR')):
			i = 0
			for word in line.split():
				if (word != 'PSNR' and word != '='):
					if (word == 'inf'):
						psnr = 150
					else:
						psnr = word
	return psnr

"""Calculate the bitwidth """
def calculateBitWidth(chromo, numVars, variableUseCount):
	bitwidth = 0
	numFs = 0
	numIs = 0
	numSs = 0
	numCs = 0
	temp = 0
	for i in range(numVars):
		temp = 0
		temp += chromo[i]
		if (chromo[i] == 3):
			temp += 3
		bitwidth += temp * variableUseCount[i]
		if (chromo[i] == 3):
			numFs = numFs + 1
		elif (chromo[i] == 2):
			numIs = numIs + 1
		elif (chromo[i] == 1):
			numSs = numSs + 1	
		elif (chromo[i] == 0):
			numCs = numCs + 1
	numFloats.append(numFs)
	numInts.append(numIs)
	numShorts.append(numSs)
	numChars.append(numCs)
		
	return bitwidth


""" Execute the kernel """
def executeKernel(chromo, numChromo, numVars, variableList, variableType, inputFileName): #numVars = len(chromo)
#print oneLevelDegraded
#	print "In exceuteKernel: " , chromo
	#output_file = open ("kernel"+str(numChromo)+".cl",'w')
	if (len(chromo) != numVars):
		print "ERROR!!!!!!!!!!!!!!!! len(chromo) != numVars"
	output_file = open ("kernel_tentative.cl",'w')
	k = 0
	for line in data:
		isRead = 0
		alreadyRead = 0
		for v in range (0, numVars):
			if (line.strip().startswith(variableList[v]) and alreadyRead == 0):
				alreadyRead = 1
				if (variableType[v] == 'float4' or variableType[v] == 'int4' or variableType[v] == 'short4'):
					if ( 'float3' in line):
						output_file.write(line.replace('float3',values3[chromo[v]]))
					elif ( '(float)' in line):
						output_file.write(line.replace('float',values[chromo[v]]))
					else:
						output_file.write(line.replace(variableType[v],values4[chromo[v]]))
		#			print "In executeKernel ", variableType[v] , " is going to be replaced ", values4[chromo[v]] 
					#output_file.write(line.replace('float4',values4[chromo[v]]))
				elif (variableType[v] == 'float' or variableType[v] == 'int' or variableType[v] == 'short'):
					output_file.write(line.replace(variableType[v],values[chromo[v]]))
				elif (variableType[v] == 'uint' or variableType[v] == 'ushort'):
					output_file.write(line.replace(variableType[v],u_values[chromo[v]]))
#				elif (variableType[v] == 'uint' or variableType[v] == 'ushort'):
#					output_file.write(line.replace(variableType[v],uvalues[chromo[v]]))
				isRead = 1
		isAmenableVarInLine = 0
		for v in range (0, numVars):
			if (variableList[v] in line): 
				isAmenableVarInLine = 1
		if (isRead == 0):
			firstWord = line.split(' ', 1)[0]
			if (firstWord.find('\t')!=-1):
				firstWord = firstWord.replace("\t", "")
			if ((line.strip().startswith('float4') or line.strip().startswith('int4') or line.strip().startswith('short4')) and isAmenableVarInLine):
				#print "chromo = ", chromo, "  len = ", len(chromo), " numVars = ", numVars
			#	print "index = ", k, " chromo = ", chromo[k], " firstWord = ", firstWord
				output_file.write(line.replace(firstWord,values4[chromo[k]]))
				k=k+1
			elif ((line.strip().startswith('float') or line.strip().startswith('int') or line.strip().startswith('short')) and isAmenableVarInLine):
				output_file.write(line.replace(firstWord,values[chromo[k]]))
				k=k+1
			elif ((line.strip().startswith('uint') or line.strip().startswith('ushort')) and isAmenableVarInLine):
				output_file.write(line.replace(firstWord,u_values[chromo[k]]))
				k=k+1
#			elif (line.strip().startswith('uint') or line.strip().startswith('ushort')):
#				output_file.write(line.replace(firstWord,uvalues[chromo[k]]))
#				k=k+1
			else:
				output_file.write(line)
	output_file.close()
#	outname = "out"+str(numChromo)+".log"
	outname = "out.log"
	cmd = 'cp kernel_tentative.cl /opt/AMDAPP/samples/opencl/cl/'
	cmd += kernelName
	cmd += '/bin/x86_64/Release/'
	cmd += kernelName
	cmd += '_Kernels.cl'
	os.system(cmd)
#	cmd = 'cd /opt/AMDAPP/samples/opencl/cl/SobelFilter'
#	os.system(cmd)
#	cmd = 'ls'
#	os.system(cmd)
	if (kernelName == 'DCT' or kernelName == 'SimpleConvolution'):
		cmd = '/opt/AMDAPP/samples/opencl/cl/'
		cmd += kernelName
		cmd += '/bin/x86_64/Release/'
		cmd += kernelName
		cmd += ' -e --inputImage '#/opt/AMDAPP/samples/opencl/cl/'
		#cmd += kernelName
		#cmd += '/Inputs/'
		cmd += inputFileName
#		cmd += kernelName
#		cmd += '_Input.bmp --kernelName '
		cmd += '.txt --kernelName '
		cmd += kernelName
		cmd += '_Kernels.cl >>' 
		cmd += outname
		os.system(cmd)
	else:		
		cmd = '/opt/AMDAPP/samples/opencl/cl/'
		cmd += kernelName
		cmd += '/bin/x86_64/Release/'
		cmd += kernelName
		cmd += ' -e --outputImage out.bmp --inputImage '
		cmd += inputFileName
		cmd += '.bmp --kernelName '
		cmd += kernelName
		cmd += '_Kernels.cl >>' 
		cmd += outname
		os.system(cmd)
#	cmd = 'rm /opt/AMDAPPSDK-3.0-0-Beta/samples/opencl/cl/SobelFilter/bin/x86/Release/kernel_*.cl'
#	os.system(cmd)
	passed = 0
#	outname = "out.log"
	input_file = open(outname, 'r')
	data1 = input_file.readlines()
	for line in data1:
		if (line.find(' nan') != -1):
			print "nan"
			passed = 0
		elif (line.strip().startswith('Passed!')):
			passed = 1
#			print "Passed!!!"
		elif (line.strip().startswith('Error:')):
			print "Build Error!!!!"
			passed = 0
		elif (line.strip().startswith('Failed!')):
			print "Failed!!!"
			passed = 0
#			print "passed"
#	if (passed == 0):
	#cmd = 'rm out.log'#
#	os.system(cmd)
#	if (passed == 1): # and numChromo != -1):
	output_file = "kernel"+str(numChromo)+".cl"
#	print "kernelName = ", output_file
	cmd = "mv kernel_tentative.cl " + output_file
 	os.system(cmd)
	output_file = "out"+str(numChromo)+".log"
#	print "outputFile = ", output_file
	cmd = "mv out.log " + output_file
 	os.system(cmd)
#	cmd = "rm out*.log"
#	os.system(cmd)

	return passed

"""Takes a population of chromosomes and returns a list of tuples where each chromo is paired to its fitness scores and ranked accroding to its fitness"""
def rankPop (chromos, numVars, variableUseCount): #numVars = len(chromos)
#	proteins, outputs, errors = [], [], []
	psnr, bitwidth = [], []
  	fitnessScores, index = [], []
	i = 1
	numChromo = 0
	for chromo in chromos: 
	#	psnr_val = readPSNR(numChromo)
	#	psnr.append(psnr_val)
		bitwidth_val = calculateBitWidth(chromo, numVars, variableUseCount)
		bitwidth.append(bitwidth_val)
		numChromo = numChromo + 1
	for i in range (0, popN):
		worstScore = 6*numVars
		score = worstScore - bitwidth[i] 
		fitnessScores.append(float(score)/worstScore)
		index.append(i)

#	print "psnr = ", psnr
#	print "bitwidth = ", bitwidth
#	print "numFloats = ", numFloats
#	print "numInts = ", numInts
#	print "numShorts = ", numShorts
#	print "numChars = ", numChars
	pairedPop = zip (chromos, bitwidth, numFloats, numInts, numShorts, numChars, fitnessScores, index)

	rankedPop = sorted(pairedPop, key = itemgetter(1, 2, 3, 4, 5))
		
	chromos = [x[0] for x in rankedPop]
	bitwidth = [x[1] for x in rankedPop]
	fitnessScores = [x[6] for x in rankedPop]
	index = [x[7] for x in rankedPop]
#	for i in range(0, numVars)
		
#	bitwidth = [x[1] for x in rankedPop]
#	print "rankedPop = ", rankedPop	
	#print "bitwidth = ", bitwidth
	#print "fitness = ", fitnessScores
	#print "index = ", index
#  	pairedPop = zip ( chromos, proteins, outputs, fitnessScores) # pair each chromo with its protein, ouput and fitness score
#  	rankedPop = sorted ( pairedPop,key = itemgetter(-1), reverse = True ) # sort the paired pop by ascending fitness score"""
	return rankedPop


""" taking a ranked population selects two of the fittest members using roulette method"""
def selectFittest (fitnessScores, rankedChromos):
	while 1 == 1: # ensure that the chromosomes selected for breeding are have different indexes in the population
		index1 = roulette (fitnessScores)
		index2 = roulette (fitnessScores)
#		print "select fitness in the while index1 = ", index1, " index2 = ", index2
    		if index1 == index2:
      			continue
    		else:
      			break

#	print "select Fitness after the while index1 = ", index1, " index2 = ", index2
	ch1 = rankedChromos[index1] # select  and return chromosomes for breeding 
	ch2 = rankedChromos[index2]
	
#	print "select Fitness ch1 ", ch1, " ch2 = ", ch2
	return ch1, ch2

"""Fitness scores are fractions, their sum = 1. Fitter chromosomes have a larger fraction.  """
def roulette (fitnessScores):
	index = 0
	cumalativeFitness = 0.0
	r = random.random()
	for i in range(len(fitnessScores)): # for each chromosome's fitness score
		
		cumalativeFitness += fitnessScores[i] # add each chromosome's fitness score to cumalative fitness

#		print "roulette -- cumalativeFitness ", cumalativeFitness, " fitnessScores[i] = ", fitnessScores[i], " r = ", r
		if cumalativeFitness > r: # in the event of cumalative fitness becoming greater than r, return index of that chromo
			return i


def crossover (ch1, ch2):
  # at a random chiasma
	r = random.randint(0,len(ch1))
	return ch1[:r]+ch2[r:], ch2[:r]+ch1[r:]


def mutate (ch):
	mutatedCh = []
	for i in ch:
		if random.random() < mutation_rate:
			newBit = random.randint(lowerBounds[i],exactBound[i])
			while newBit == i:
				newBit = random.randint(lowerBounds[i],exactBound[i])
			mutatedCh.append(newBit)
#			print "mutation on ", mutatedCh  
    		else:
      			mutatedCh.append(i)
  #assert mutatedCh != ch
	return mutatedCh
 
def isChromGood(newnewCh1, ind, bestFound, numVars, variableList, variableType, inputFileName): 
	isAcc = accuracyCheck(newnewCh1, bestFound, numVars) # isAcc = -1 : can't decide w/o execution, = 0: can decide/not accurate, = 1: can decide/accurate
	if (isAcc == -1):		 
		passed = executeKernel(newnewCh1, ind, numVars, variableList, variableType, inputFileName)
	elif (isAcc == 1):
		passed = 1
		#print "WOW: no execution, passed! bestFound = ",bestFound  
	elif (isAcc == 0):
		passed = 0
		#print "WOW: no execution, failed!"
	if (passed == 1): 
		#print numChromo, chromo
		bestFound = updateBestFound(bestFound, newnewCh1) 
	#	if (isAcc == -1):
			#print "isChromGood: chrom after executing is = ", newnewCh1
	return passed

"""Using breed and mutate it generates two new chromos from the selected pair"""
def breed (ch1, ch2, ind, bestFound, numVars, variableList, variableType, inputFileName):
	newCh1, newCh2 = [], []
  	if random.random() < crossover_rate: # rate dependent crossover of selected chromosomes
    		newCh1, newCh2 = crossover(ch1, ch2)	
  	else:
    		newCh1, newCh2 = ch1, ch2
  	newnewCh1 = mutate(newCh1) # mutate crossovered chromos
	passed = isChromGood(newnewCh1, ind, bestFound, numVars, variableList, variableType, inputFileName)
  	if (passed == 0):
		#while (passed == 0):
		newnewCh1 = mutate (newCh1) # mutate crossovered chromos
		passed = isChromGood(newnewCh1, ind, bestFound,numVars, variableList, variableType, inputFileName)
#  		passed = executeKernel(newnewCh1, ind, len(newnewCh1))
#	print "newnewCh1 ", newnewCh1, " passed = ", passed
  	newnewCh2 = mutate (newCh2) 
#  	passed1 = executeKernel(newnewCh2, ind+1, len(newCh2))
	passed1 = isChromGood(newnewCh2, ind+1, bestFound, numVars, variableList, variableType, inputFileName)
  	if (passed1 == 0):
		#while (passed == 0):
  		newnewCh2 = mutate (newCh2) 
 # 		passed1 = executeKernel(newnewCh2, ind+1, len(newnewCh2))
		passed1 = isChromGood(newnewCh2, ind+1, bestFound, numVars, variableList, variableType, inputFileName)
#	print "newnewCh2 ", newnewCh2, " passed = ", passed1
  	return newnewCh1, newnewCh2, passed, passed1

""" Taking a ranked population return a new population by breeding the ranked one"""
def iteratePop (rankedPop, bestFound, numVars, variableList, variableType, inputFileName):
	fitnessScores = [ item[6] for item in rankedPop ] # extract fitness scores from ranked population
	rankedChromos = [ item[0] for item in rankedPop ] # extract chromosomes from ranked population
	
#	print "iteratePop fitnessScores= ", fitnessScores
	newpop = []
	newpop.extend(rankedChromos[:popN/ELITISM_RATE]) # known as elitism, conserve the best solutions to new population
#	print "elitisim for ", popN/ELITISM_RATE, " items"
#	for i in range(0, len(newpop)):
#		print newpop[i]
#	print "breeding for the rest"
	while len(newpop) != popN:
		ch1, ch2 = [], []
#		print "before selectFittest fitnessScore = " , fitnessScores
		ch1, ch2 = selectFittest (fitnessScores, rankedChromos) # select two of the fittest chromos
#        	print "it selectes : ", ch1, "  ",  ch2
		i = len(newpop) 
#		print "before breed"
    		ch1, ch2, passed, passed1 = breed(ch1, ch2, i, bestFound, numVars, variableList, variableType, inputFileName) # breed them to create two new chromosomes 
#		print "after breed ch1 = ", ch1, " ch2\ = ", ch2
		if (passed == 1 and len(newpop) != popN):
    			newpop.append(ch1) # and append to new population
#			print "new chrom1 = ", ch1
    		if (passed1 == 1 and len(newpop) != popN):
			newpop.append(ch2)
#			print "new chrom2 = ", ch2
		
  	return newpop
  
      
def configure (variableList, variableType, variableUseCount):
	N = 0
	global vals
	global input_file
	global data
	vals = ['char','short','int','float']
	#tolerableVariables = []
	global line

	input_file = open("kernel.cl", 'r')

	data = input_file.readlines()
	
	for line in data:
	        isNext = 0;
		ifIsNext = 0;
		if (line.strip().startswith('float4') or line.strip().startswith('int4') or line.strip().startswith('short4') or line.strip().startswith('float') or line.strip().startswith('int') or line.strip().startswith('short') or line.strip().startswith('float3') or line.strip().startswith('int3') or line.strip().startswith('short3') or line.strip().startswith('uint') or line.strip().startswith('ushort') ):
			#print line.strip()
			for word in line.split():
				if (ifIsNext == 1 and word == ','):
					isNext = 1
				if (isNext == 1 and word != ','): 
					N = N + 1
					isNext = 0
					ifIsNext = 1
					if (word.find(';')!=-1): #the string contains semicolon
						word = word.replace(";", "")
					if (word.find(',')!=-1): #the string contains semicolon
						word = word.replace(",", "")
						isNext = 1
					variableList.append(word)
					firstWord = line.split(' ', 1)[0]
#					print "firstWord before: ", firstWord
					if (firstWord.find('\t')!=-1):
						firstWord = firstWord.replace("\t", "")
					
#					print "firstWord after: ", firstWord
					variableType.append(firstWord)
				if (word=='float4' or word == 'int4' or word == 'short4' or word == 'float' or word == 'int' or word == 'short' or word == 'uint' or word == 'ushort'): #or word=='float3' or word == 'int3' or word == 'short3' 
					isNext = 1
	for i in range (0, N):
		variableUseCount.append(-1) # -1 for definition

	for line in data:
		for v in variableList:
			if (line.find(v) != -1):
				variableUseCount[variableList.index(v)] += line.count(v)
			if (line.strip().startswith(v)):	
				variableUseCount[variableList.index(v)] -= 1 #if it is a write, I want to approximately consider the effect on operations
				
	print N
	print vals[0]           
	print variableList
	print variableType
	print ">>>>>>>>> variableCount = ", variableUseCount
	return N, variableList, variableType, variableUseCount    


def findAmenableVars(numVars,variableList, variableType, inputFileName):
	oneLevelDegraded = [3 for i in xrange(numVars)]
	passed = []
	for i in range (0, numVars):
		for j in range (0, numVars):
			if (i == j):
				if (variableType[j] in values4):
					oneLevelDegraded[j] = values4.index(variableType[j])-1
				elif (variableType[j] in values):
					oneLevelDegraded[j] = values.index(variableType[j])-1
				elif (variableType[j] in u_values):
					oneLevelDegraded[j] = u_values.index(variableType[j])-1
				#oneLevelDegraded[i][j] = values4[values4.index(variableType[j])-1]
			else:
 				if (variableType[j] in values4):
					oneLevelDegraded[j] = values4.index(variableType[j])
				elif (variableType[j] in values):
					oneLevelDegraded[j] = values.index(variableType[j])	
				elif (variableType[j] in u_values):
					oneLevelDegraded[j] = u_values.index(variableType[j])
		#passed = verifyKernel(oneLevelDegraded, numVars)
		#print "finding Amenable Vars for ", oneLevelDegraded, " varList : ", variableList
		passed.append(executeKernel(oneLevelDegraded,i, numVars, variableList, variableType, inputFileName))
		print "in findAmenable i = ", i, "chrom: ", oneLevelDegraded, " : passed = ", passed	
	return passed
			
def pruneVariables(isPassed, variableList, variableType, variableUseCount):
	toNotPrunedType = []
	toNotPrunedList = []
	toNotPrunedCount = []
	numVars = len(isPassed)
#	print variableType
	for i in range (0, numVars):
		if (isPassed[i] != 0):
			toNotPrunedType.append(variableType[i])
			toNotPrunedList.append(variableList[i])
			toNotPrunedCount.append(variableUseCount[i])
	variableType = []
	variableList = []
	variableUseCount = []

	while (len(variableType) > 0):
#		print "index = ", i
		variableList.pop()
		variableType.pop()
		variableUseCount.pop()

	numVars = len(toNotPrunedList)
	print "numVars = ", numVars	
	for i in range (0, numVars):
		variableList.append(toNotPrunedList[i])
		variableType.append(toNotPrunedType[i])
		variableUseCount.append(toNotPrunedCount[i])
	#print "after pruning : variableList = ", variableList
	#print "after pruning: variableType = ", variableType
	#print "after pruning: variableUseCount = ", variableUseCount

#	print " length = ", len(variableType)
	if (numVars != len(variableType)):
		print ">>>>>>> Error in prunVariables!! "
		#	variableType.pop(i)
		#	variableList.pop(i)
		#	numVars -= 1
	return numVars, variableList, variableType, variableUseCount

def findLowerBounds(numVars, variableList, variableType, inputFileName):
	global lowerBounds
	lowerBounds = [0 for i in xrange(numVars)]
	degradedList = [0 for i in xrange(numVars)]
	
	i = 0
	level = 0
	while i < numVars:
		for j in range(numVars):
			if (i == j):
				degradedList[j] = level
				#oneLevelDegraded[i][j] = values4[values4.index(variableType[j])-1]
			else:
 				if (variableType[j] in values4):
					degradedList[j] = values4.index(variableType[j])
				elif (variableType[j] in values):
					degradedList[j] = values.index(variableType[j])	
				elif (variableType[j] in u_values):
					degradedList[j] = u_values.index(variableType[j])
		#passed = verifyKernel(oneLevelDegraded, numVars)
		isPass = executeKernel(degradedList,-1, numVars, variableList, variableType, inputFileName)
		if (isPass == 0):
			level += 1
		else: 
			lowerBounds[i] = level
			i += 1
			level = 0
# here we try the lowerbound itself to see if it passes, if passes it would be the best possible kernel
#	print "Executing the all lowerBounds ..."
	isPass = executeKernel(lowerBounds,-1, numVars, variableList, variableType, inputFileName)
	print "After executing the all lowerBounds ..."
	if (numVars == 0):
		print "number of amenable vars are 0!!!!"
	print "lower Bounds:", lowerBounds, " isPass = ", isPass
	return isPass
		
def initializeExactBound(numVars, variableType):
	global exactBound
	exactBound = []
	for i in range(numVars):
		if (variableType[i]=='float4' or variableType[i] == 'float' or variableType[i] == 'float3'):
			exactBound.append(3)
		elif (variableType[i]=='int4' or variableType[i] == 'int' or variableType[i] == 'uint'): # or variableType[i] == 'int3'):
			exactBound.append(2)
		elif (variableType[i]=='short4' or variableType[i] == 'short' or variableType[i] == 'ushort'): #or variableType[i] == 'short3'):
			exactBound.append(1)
		elif (variableType[i]=='char4' or variableType[i] == 'char' or variableType[i] == 'uchar'): #variableType[i] == 'char3'):
			exactBound.append(0)
#	print "exactBoundLen = ", len(exactBound), " numVars = ", numVars 

""" generate final kernels """
def generateKernel(chromo, numChromo, numVars, variableList, variableType, inputFileName): #numVars = len(chromo)
#print oneLevelDegraded
#	print "In exceuteKernel: " % chromo
	output_file = open ("kernel_tentative.cl",'w')
	k = 0
#	print "generateKernel: ", chromo
#	print  "generateKernel: ", variableList
#	print  "generateKernel: ",  variableType
	for line in data:
		isRead = 0
		alreadyRead = 0
		for v in range (0, numVars):
			if (line.strip().startswith(variableList[v]) and alreadyRead == 0):
				alreadyRead = 1
				if (variableType[v] == 'float4' or variableType[v] == 'int4' or variableType[v] == 'short4'):
					if ( 'float3' in line):
						output_file.write(line.replace('float3',values3[chromo[v]]))
					elif ( '(float)' in line):
						output_file.write(line.replace('float',values[chromo[v]]))
					else:
						output_file.write(line.replace(variableType[v],values4[chromo[v]]))
				#	print line, "In executeKernel ", variableType[v] , " is going to be replaced ", values4[chromo[v]] 
					#output_file.write(line.replace('float4',values4[chromo[v]]))
				if (variableType[v] == 'float' or variableType[v] == 'int' or variableType[v] == 'short'):
					output_file.write(line.replace(variableType[v],values[chromo[v]]))
				if (variableType[v] == 'uint' or variableType[v] == 'ushort'):
					output_file.write(line.replace(variableType[v],u_values[chromo[v]]))
				#	print line, "In executeKernel ", variableType[v] , " is going to be replaced ", values[chromo[v]] 
#				elif (variableType[v] == 'uint' or variableType[v] == 'ushort'):
#					output_file.write(line.replace(variableType[v],uvalues[chromo[v]]))
				isRead = 1
		isAmenableVarInLine = 0
		for v in range (0, numVars):
			if (variableList[v] in line): 
				isAmenableVarInLine = 1
		if (isRead == 0):
			firstWord = line.split(' ', 1)[0]
			if (firstWord.find('\t')!=-1):
				firstWord = firstWord.replace("\t", "")
			if ((line.strip().startswith('float4') or line.strip().startswith('int4') or line.strip().startswith('short4')) and isAmenableVarInLine):
				output_file.write(line.replace(firstWord,values4[chromo[k]]))
				k=k+1
			elif ((line.strip().startswith('float') or line.strip().startswith('int') or line.strip().startswith('short')) and isAmenableVarInLine):
				output_file.write(line.replace(firstWord,values[chromo[k]]))
				k=k+1
			elif ((line.strip().startswith('uint') or line.strip().startswith('ushort')) and isAmenableVarInLine):
				output_file.write(line.replace(firstWord,u_values[chromo[k]]))
				k=k+1

#			elif (line.strip().startswith('uint') or line.strip().startswith('ushort')):
#				output_file.write(line.replace(firstWord,uvalues[chromo[k]]))
#				k=k+1
			else:
				output_file.write(line)
	output_file.close()
#	outname = "out"+str(numChromo)+".log"
	outname = "out_final.log"

	cmd = 'cp kernel_tentative.cl /opt/AMDAPP/samples/opencl/cl/'
	cmd += kernelName
	cmd += '/bin/x86_64/Release/'
	cmd += kernelName
	cmd += '_Kernels.cl'
	os.system(cmd)
	if (kernelName == 'DCT' or kernelName == 'SimpleConvolution'):
		cmd = '/opt/AMDAPP/samples/opencl/cl/'
		cmd += kernelName
		cmd += '/bin/x86_64/Release/'
		cmd += kernelName
		cmd += ' -e --inputImage '#/opt/AMDAPP/samples/opencl/cl/'
		#cmd += kernelName
		#cmd += '/Inputs/'
		cmd += inputFileName
#		cmd += kernelName
#		cmd += '_Input.bmp --kernelName '
		cmd += '.txt --kernelName '
		cmd += kernelName
		cmd += '_Kernels.cl >>' 
		cmd += outname
		os.system(cmd)

	else:		
		cmd = '/opt/AMDAPP/samples/opencl/cl/'
		cmd += kernelName
		cmd += '/bin/x86_64/Release/'
		cmd += kernelName
		cmd += ' -e --outputImage out.bmp --inputImage '
		cmd += inputFileName
		cmd += '.bmp --kernelName '
		cmd += kernelName
		cmd += '_Kernels.cl >>' 
		cmd += outname
		os.system(cmd)
	
	passed = 0
	input_file = open(outname, 'r')
	data1 = input_file.readlines()
	for line in data1:
		if (line.strip().startswith('Passed!')):
			passed = 1
		if (line.strip().startswith('Failed')):
			passed = 0
		if (line.strip().startswith('Error:')):
			print "Build Error!!!!! "
#			print "Crazy things are happening: Passed and Failed at the same time!!!"

#			print "passed"
	if (passed == 0):
		print "WHAT????? The final set should all pass!!!! for chromo = ", chromo
	output_file = "kernel"+str(numChromo)+".cl"
	cmd = "mv kernel_tentative.cl " + output_file
	os.system(cmd)
	output_file = "out"+str(numChromo)+".log"
	cmd = "mv out_final.log " + output_file
	os.system(cmd)
	return passed


def testOtherInputs(chromos, numVars, variableList, variableType, inputFileName, islowerPassed): #if the best Found failed, try others in the generation, otherwise do the GA again
	i = int(inputFileName)
	noway = 0
	j = 0
	while i < 100:
		inputFileName = str(i)
		print "j = ", j, " chromos[j] = ", chromos[j]
		passed = executeKernel(chromos[j], -1, numVars, variableList, variableType, inputFileName)
		if (passed == 1):
			print "test ", i, " in the training set passed! with chromo [", j, "], chromo = ", chromos[j]
			i = i + 1
		elif (passed == 0 and len(chromos) > j+1): #islowerPassed == 0): #meaning a GA is run beforehand	
			j = j + 1
			if j >= popN:
				passedAll = 0
				noway = 1
				i = 100
				print "test ", inputFileName, " in testing other inputs: failed and should do the GA : chromo = ", chromos[j]
		elif (passed == 0):
			passedAll = 0
			print "test ", i, " in the training set failed! with chromo [", j, "], chromo = ", chromos[j]
			noway = 1
			i = 100
	if ( passed == 1 and noway == 0 ):
		passedAll = 1
		inputFileName = str(100)
	return passedAll, inputFileName

"""		elif (passed == 0 and islowerPassed == 1):
			passedAll = 0
			noway = 1
			i = 100
			print "test ", inputFileName, " in testing other inputs: failed and should do the GA : chromo = ", chromos[j]
"""

 
def main():
	#print 'Number of Arguments: ', len(sys.argv), ' arguments.'
	global kernelName
	kernelName = str(sys.argv[1])
	print 'Kernel Name: ', kernelName
	inputFileName = '1'
	isPassed = []
	variableList = []
	variableType = []
	variableUseCount = [] 
	bestFitnessScore = 0
	noImprovement = 0
	numVars, variableList, variableType, variableUseCount = configure(variableList, variableType, variableUseCount)
	
	chromos = []
	toBeContinued = 1
	#print "after configure: type = ", variableType
#	while toBeContinued == 1: 
	isPassed = findAmenableVars(numVars, variableList, variableType, inputFileName)
		#print "isPassed = ", isPassed
	numVars, variableList, variableType, variableUseCount = pruneVariables(isPassed, variableList, variableType, variableUseCount)
	initializeExactBound(numVars, variableType)
#		print "in main after calling pruneVar : ",  variableList
#		print  "in main after calling pruneVar : ",   variableType
#		print  "in main after calling pruneVar : ",   variableUseCount

	islowerPassed = findLowerBounds(numVars, variableList, variableType, inputFileName)
	while toBeContinued == 1: 
		print "input file name for the islowerPassed = " , inputFileName

		if (islowerPassed == 0):
			chromos = []
			print "the lowerBound kernel is failed!"
			print "APPLYING GA ..... "
			chromos = generatePop(numVars, bestFound, variableList, variableType, inputFileName) #generate new population of random chromosomes
			iterations = 0
			solution_found = False
			rankedPop = rankPop(chromos, numVars, variableUseCount) 
			while iterations != max_iterations and solution_found != True:
				print "Iteration : ", iterations
    				# take the pop of random chromos and rank them based on their fitness score/proximity to target output
 			   	#print '\nCurrent iterations:', iterations
    	
	#if solution_found != True:
      # if solution is not found iterate a new population from previous ranked population
				chromos = iteratePop(rankedPop, bestFound, numVars, variableList, variableType, inputFileName)
#				print "chromos =  ", chromos      
				rankedPop = rankPop(chromos, numVars, variableUseCount)  #to be removed
				fitnessScores = [x[6] for x in rankedPop]
				if fitnessScores[0] > bestFitnessScore:
					bestFitnessScore = fitnessScores[0]
				elif fitnessScores[0]== bestFitnessScore:
					noImprovement += 1
				if noImprovement == 10:
					solution_found = True 
					print "solution is converged after ", iterations, " iterations"
				iterations += 1
			print "bestFound = ", bestFound
			for i in range(popN):
				generateKernel(chromos[i], i, numVars, variableList, variableType, inputFileName) #numVars = len(chromo)
		if (islowerPassed == 1):
			generateKernel(lowerBounds, 0, numVars, variableList, variableType, inputFileName) #numVars = len(chromo)
			print "without running GA final choromo = ", lowerBounds
			chromos.append(lowerBounds)
			for i in range (1,len(chromos)):
				if (chromos[0] == chromos[i]):
					print "there is a problem in the chromo generation!"
			print ">>>>>>>>>>>>>>>>>>>.. final choromo = ", chromos
		passedAll, inputFileName = testOtherInputs(chromos, numVars, variableList, variableType, inputFileName, islowerPassed) #if the best Found failed, try others in the generation, otherwise do the GA again
		if (passedAll == 0):
			print ">>>>>>>>>>>>>>>> Input file ", str(inputFileName), " failed!"
			islowerPassed = 0
		if (passedAll == 1):
			toBeContinued = 0
	
		print ">>>>> CHROMROS = ", chromos
	input_file.close()
#    else:
#      break


if __name__ == "__main__":
    main()
