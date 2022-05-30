#!/usr/bin/env python3
import csv
import flops_computing

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101
from torchvision.models import vgg16, vgg13, vgg11, alexnet, squeezenet1_0, squeezenet1_1

sqnt10=squeezenet1_0
sqnt11=squeezenet1_1

def get_flops(name,dDict):
	if name in dDict:
		result=dDict[name]
	else:
		result=flops_computing.get_list(eval(name+"()"))
		dDict[name]=result

	return result

def calc_flops(lFlops,sRange):
	if sRange=="full":
		return sum(lFlops)
	else:
		iMin,iMax=eval(sRange)
		return sum(lFlops[iMin-1:iMax-1])

def parse_line(lLine,lFields,dFlops):
	flops1=None
	flops2=None
	range1=None
	range2=None

	for i in range(len(lFields)):
		if lFields[i]=="model_1":
			flops1=get_flops(lLine[i],dFlops)
		elif lFields[i]=="model_2":
			flops2=get_flops(lLine[i],dFlops)
		elif lFields[i]=="op_range":
			if range1 is None:
				range1=lLine[i]
			elif range2 is None:
				range2=lLine[i]
			else:
				assert False

	lLine.append(calc_flops(flops1,range1))
	lLine.append(calc_flops(flops2,range2))

def main():
	dFlops = dict()
	with open("sampled_data.csv","r") as fFile1:
		with open("sampled_data_processed.csv","w") as fFile2:
			rReader=csv.reader(fFile1)
			wWriter=csv.writer(fFile2)

			lFields=next(rReader)
			wWriter.writerow(lFields+["flops_1","flops_2"])
			for lLine in rReader:
				parse_line(lLine,lFields,dFlops)
				wWriter.writerow(lLine)

if __name__=="__main__":
	main()
