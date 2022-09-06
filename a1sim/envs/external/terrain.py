#! /usr/bin/python3

import pickle
import random
import numpy as np
from PIL import Image
from os import path

TERRAIN_CHOICE = []
CURRICULUM_CHOICE = {}


TERRAIN_PLANE = 0
TERRAIN_TRAINING = 1
TERRAIN_VALIDATION = 2
TERRAIN_TEST = 3
TERRAIN_ICE = 4
TERRAIN_SAMURAI = 5
TERRAIN_BLOBBED = 6

OBJECT_EMPTY = 0
OBJECT_BALLS = 1

PATH_SRC    = path.dirname(path.realpath(__file__))
PATH_ROOT   = path.dirname(PATH_SRC)
PATH_DATA   = PATH_ROOT+"/data"

SUBPATH_TERRAIN = { TERRAIN_SAMURAI:   "terrains/samurai/samurai.urdf",
                    TERRAIN_BLOBBED:   "terrains/blobbed_terrain/terrain.urdf"}

PATH_TERRAIN_TRAINING=PATH_DATA+'/dataset'
PATH_TERRAIN_VALIDATION=PATH_DATA+'/dataset'


def loadTerrain(path):

    import sys
    import os

    global TERRAIN_CHOICE

    if len(TERRAIN_CHOICE) == 0:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith("pkl"):
                    TERRAIN_CHOICE.append(os.path.join(root,file))

    if len(CURRICULUM_CHOICE) == 0:
        for pathType in os.listdir(path):
            for pathCur in os.listdir("{}/{}".format(path,pathType)):
                cur = int(pathCur)

                if cur not in CURRICULUM_CHOICE.keys():
                    CURRICULUM_CHOICE[cur] = []

                for root, dirs, files in os.walk("{}/{}/{}".format(path, pathType, pathCur)):
                    for file in files:
                        if file.endswith("pkl"):
                            CURRICULUM_CHOICE[cur].append(os.path.join(root,file))

    return True


# Define objects for the map ##
class modelTerrain():

    def __init__(self, pybullet, typeTerrain=TERRAIN_PLANE, curriculum=None):

        self.sim = pybullet
        self.typeTerrain = typeTerrain
        self.measurement = None
        self.curriculum = curriculum

        if self.typeTerrain in [TERRAIN_SAMURAI, TERRAIN_BLOBBED]:
            offsetHeight = 0
            self.idTerrain = self.sim.loadURDF(SUBPATH_TERRAIN[typeTerrain], [0.0, 0.0, 0.0], useFixedBase=1)

            self.sim.resetBasePositionAndOrientation(self.idTerrain,[0 ,0,offsetHeight], [0,0,0,1])
            self.sim.changeDynamics(self.idTerrain, -1, lateralFriction=1.0)

        elif self.typeTerrain == TERRAIN_TEST:
            self.numHeightfieldRows = 256
            self.numHeightfieldColumns = 256
            self.raHeightField = [[0 for j in range(self.numHeightfieldRows)] for i in range (self.numHeightfieldColumns)]

            heightPerturbationRange = random.uniform(0,0.1) # 0.06
            
            for i in range (int(self.numHeightfieldColumns/2)):
                for j in range (int(self.numHeightfieldRows/2) ):
                    height = random.uniform(0,heightPerturbationRange)
                    self.raHeightField[2*i][2*j]=height
                    self.raHeightField[2*i+1][2*j]=height
                    self.raHeightField[2*i][2*j+1]=height
                    self.raHeightField[2*i+1][2*j+1]=height
            
            raFlattenHeightField = []
            for raColumnHeightField in self.raHeightField:
                raFlattenHeightField += raColumnHeightField

            # offsetHeight = 0.5*(max(raFlattenHeightField)+min(raFlattenHeightField)) - max([0,heightPerturbationRange -0.04])
            # offsetHeight = 0.5*(max(raFlattenHeightField)+min(raFlattenHeightField) - heightPerturbationRange)
            offsetHeight = 0

            shapeTerrain = self.sim.createCollisionShape(shapeType = self.sim.GEOM_HEIGHTFIELD, meshScale=[.05,.05,1], heightfieldTextureScaling=1, heightfieldData=raFlattenHeightField, numHeightfieldRows=self.numHeightfieldRows, numHeightfieldColumns=self.numHeightfieldColumns)
            self.idTerrain  = self.sim.createMultiBody(0, shapeTerrain)

            self.sim.resetBasePositionAndOrientation(self.idTerrain,[6 ,0,offsetHeight], [0,0,0,1])
            self.sim.changeDynamics(self.idTerrain, -1, lateralFriction=1.0)

        elif self.typeTerrain == TERRAIN_ICE:

            offsetHeight = 0

            shapePlane = self.sim.createCollisionShape(shapeType = self.sim.GEOM_PLANE)

            raSizeBox = [0.15, 0.15 , 0.001]
            raShiftAng = [0, 0, 0, 1]
            raColorRGB = [0, 1, 1, 1]
            raColorSpecular = [0.4, .4, 0]

            idVisualShape = self.sim.createVisualShape( shapeType=self.sim.GEOM_BOX, halfExtents=raSizeBox, 
                                                        rgbaColor=raColorRGB, specularColor=raColorSpecular)
            idCollisionShape = self.sim.createCollisionShape(shapeType=self.sim.GEOM_BOX, halfExtents=raSizeBox)

            for i in range(5):
                for j in range(5):

                    idIce = self.sim.createMultiBody(0, idCollisionShape, idVisualShape, [i -2, j-2, 0.0005], raShiftAng)
                    self.sim.changeDynamics(idIce, -1, lateralFriction=0.01)

            self.idTerrain  = self.sim.createMultiBody(0, shapePlane)

            self.sim.resetBasePositionAndOrientation(self.idTerrain,[0 ,0,offsetHeight], [0,0,0,1])
            self.sim.changeDynamics(self.idTerrain, -1, lateralFriction=1.0)

        elif self.typeTerrain in [TERRAIN_TRAINING, TERRAIN_VALIDATION]:


            if self.curriculum == None:
                path  = random.choice(TERRAIN_CHOICE)

            else:
                if self.curriculum in CURRICULUM_CHOICE.keys():
                    path  = random.choice(CURRICULUM_CHOICE[self.curriculum])
                else:
                    path  = random.choice(TERRAIN_CHOICE)

            # path = TERRAIN_CHOICE[115]

            with open(path, "rb") as f:
                choice = pickle.load(f)

            self.raHeightField = choice['height']
            self.raCostField = choice['cost']
            self.numHeightfieldRows = len(self.raHeightField[0])
            self.numHeightfieldColumns = len(self.raHeightField)
            self.raMeshScale = [.05,.05, 1]

            raFlattenHeightField = []
            for raColumnHeightField in self.raHeightField:
                raFlattenHeightField += raColumnHeightField

            offsetHeight = 0.5 * (np.max(raFlattenHeightField) + np.min(raFlattenHeightField))
            # offsetHeight = np.average(raFlattenHeightField)
            self.shapeTerrain = self.sim.createCollisionShape(shapeType = self.sim.GEOM_HEIGHTFIELD, meshScale=self.raMeshScale, heightfieldTextureScaling=1, heightfieldData=raFlattenHeightField, numHeightfieldRows=self.numHeightfieldRows, numHeightfieldColumns=self.numHeightfieldColumns)
            self.idTerrain  = self.sim.createMultiBody(0, self.shapeTerrain)

            self.sim.resetBasePositionAndOrientation(self.idTerrain,[0 ,0,offsetHeight], [0,0,0,1])
            self.sim.changeDynamics(self.idTerrain, -1, lateralFriction=1.0)

        else:
            offsetHeight = 0
            shapePlane = self.sim.createCollisionShape(shapeType = self.sim.GEOM_PLANE)
            self.idTerrain  = self.sim.createMultiBody(0, shapePlane)

            self.sim.resetBasePositionAndOrientation(self.idTerrain,[0 ,0,offsetHeight], [0,0,0,1])
            self.sim.changeDynamics(self.idTerrain, -1, lateralFriction=1.0)


    def resetTerrain(self):

        if self.typeTerrain in [TERRAIN_TRAINING, TERRAIN_VALIDATION]:

            self.raHeightField, self.raCostField = random.choice(TERRAIN_CHOICE)
            self.numHeightfieldRows = len(self.raHeightField[0])
            self.numHeightfieldColumns = len(self.raHeightField)
            self.raMeshScale = [.02,.02,1]

            raFlattenHeightField = []
            for raColumnHeightField in self.raHeightField:
                raFlattenHeightField += raColumnHeightField

            offsetHeight = np.average(raFlattenHeightField)
            self.sim.createCollisionShape(shapeType = self.sim.GEOM_HEIGHTFIELD, meshScale=self.raMeshScale, heightfieldTextureScaling=1, heightfieldData=raFlattenHeightField, numHeightfieldRows=self.numHeightfieldRows, numHeightfieldColumns=self.numHeightfieldColumns, replaceHeightfieldIndex = self.shapeTerrain)

            self.sim.resetBasePositionAndOrientation(self.idTerrain,[6,0, offsetHeight], [0,0,0,1])

        return



class heightmap():

    def __init__(self, raHeightField, raCostField, valResRef, valResOut, raOutRange):

        self.matHeightField = np.array(raHeightField)
        self.matCostField = np.array(raCostField)
        self.valResRef = valResRef
        self.valResOut = valResOut
        self.raOutRange = raOutRange
        self.valMapOffset = 6.0

        self.numColHeightField = len(raHeightField)
        self.numRowHeightField =  len(raHeightField[0])

        self.numColOut = int(self.raOutRange[1]/self.valResOut)+1
        self.numRowOut = int(self.raOutRange[0]/self.valResOut)+1


    def sampleTerrain(self, raPos, valYaw):

        idxRowOffset = int((raPos[0] - self.valMapOffset)/self.valResOut) + int(0.5*(self.numRowHeightField-1)) - int(0.5*(self.numRowOut-1))
        idxColOffset = int(raPos[1]/self.valResOut) + int(0.5*(self.numColHeightField-1)) - int(0.5*(self.numColOut-1))

        idxColStart = max(idxColOffset, 0)
        idxColTerm = min(idxColOffset + self.numColOut, self.numColHeightField)

        idxRowStart = max(idxRowOffset, 0)
        idxRowTerm = min(idxRowOffset + self.numRowOut, self.numRowHeightField)

        raHeightMap = np.zeros((self.numColOut, self.numRowOut))
        raHeightMap[idxColStart- idxColOffset:idxColTerm - idxColOffset,idxRowStart- idxRowOffset:idxRowTerm- idxRowOffset] = self.matHeightField[idxColStart:idxColTerm,idxRowStart:idxRowTerm]

        return raHeightMap.tolist()


    def checkAffordance(self, raPos):

        idxRow = int((raPos[0] - self.valMapOffset)/self.valResOut) + int(0.5*(self.numRowHeightField-1))
        idxCol = int(raPos[1]/self.valResOut) + int(0.5*(self.numColHeightField-1))

        return self.matCostField[idxCol,idxRow]
