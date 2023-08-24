


def mbsEditor(
    x, 
    bodyLengthB, 
    bodyLengthS, 
    sustainerLaunchWt, 
    nozzleDiameter, 
    sustainerCG,
    boosterLaunchWt,
    boosterSeperationDelay,
    boosterCG,
    sustainerIgnitionDelay
    ):

    rocketDict = {
        'noseConeLength': x["diameter"]*6,
        'diameter': x["diameter"],
        'noseConeShape': 'Von Karman Ogive',
        'boosterLength': bodyLengthB,
        'bodyTubeDiameter': x["diameter"],
        'bodyTubeColor': 'Black',
        'sustainerLength': bodyLengthS,
        'bodyTubeDiameter2': x["diameter"],
        'sustainerRoot': x["root_2"],
        'sustainerSpan': x["span_2"],
        'sustainerSweep': x["sweep_2"],
        'sustainerTip': x["tip_2"],
        'sustainerFinLocation': x["root_2"] + 1.5,
        'boosterLength': bodyLengthB,
        'boosterRoot': x["root_1"],
        'boosterSpan': x["span_1"],
        'boosterSweep': x["sweep_1"],
        'boosterTip': x["tip_1"],
        'boosterDiameter': x["diameter"],
        'boosterLocation': x["diameter"]*6 + bodyLengthS,
        'boosterFinLocation': x["root_1"] + 1.5,
        'altitude': 2050,
        'rodLength': 24,
        'windSpeed': 0,
        'sustainerLaunchWt': sustainerLaunchWt,
        'nozzleDiameter': nozzleDiameter,
        'sustainerCG': sustainerCG,
        'boosterLaunchWt': boosterLaunchWt,
        'boosterSeperationDelay': boosterSeperationDelay,
        'boosterCG': boosterCG,
        'sustainerIgnitionDelay': sustainerIgnitionDelay
    }



    mbsFile2 = open(r'.\MBSTemplate3.CDX1', 'r')

    lineList = mbsFile2.readlines()

    for (i, line) in enumerate(lineList):
        if '$' not in line:
            continue
        split = line.split('$')
        key = split[1]
        split[1] = str(rocketDict[key])
        lineList[i] = ''.join(split)



    mbsFile2 = open(r'FS_PRO_rocket.CDX1', 'w')
    mbsFile2.writelines(lineList)
    mbsFile2.close()
    