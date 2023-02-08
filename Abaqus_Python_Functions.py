import time
import os
import random
from abaqus import *
from abaqusConstants import *
import numpy as np
import regionToolset
import __main__
from section import * 
import part
import material
import assembly
import step
import interaction
import load
import mesh
import job
import sketch
import visualization
import xyPlot
import connectorBehavior
import odbAccess
import matplotlib.pyplot as plt
from matplotlib import cm

# Functions

def Create_Random_Geometric_Imperfection(lon, cir, t, length, radius, model):    
    circle = 2.0 * np.pi * radius
    
    # Create random geometric imperfection
    mu, sigma = 0.5, 0.2
    z = np.random.normal(mu, sigma, size = [cir, lon+1]) # Random deviation
    
    # Mesh gird
    x = np.linspace(0.0,length, lon+1)
    y = np.linspace(0.0,circle,cir)
    y = np.transpose(y)
    
    X, Y = np.meshgrid(x,y)
    
    #----------------------------------------------------------------------#
    
    # A00 and B00
    fc_x = 10
    fc_y = 20
    
    A = np.zeros((fc_x+1,fc_y+1))
    B = np.zeros((fc_x+1,fc_y+1))
    A[0,0] = (1.0/(4.0*length*circle*t)) * np.trapz(np.trapz(z, y, axis = 0), x, axis = 0)
    B[0,0] = 0.0
    
    # A1l-Akl and B1l to Bkl
    for k in range(fc_x):
        for j in range(fc_y):
            fun2a = (z * np.cos((k+1)*np.pi*X/length) * np.cos((j+1)*Y/radius))
            fun2b = (z * np.cos((k+1)*np.pi*X/length) * np.sin((j+1)*Y/radius))
            
            A[k+1,j+1] = (1/(length*circle*t)) * np.trapz(np.trapz(fun2a, y, axis = 0), x, axis = 0)
            B[k+1,j+1] = (1/(length*circle*t)) * np.trapz(np.trapz(fun2b, y, axis = 0), x, axis = 0)
           
    k = 0
    j = 0
    
    # A01-A0l and B01-B0l
    for m in range(fc_y):
        fun3a = z*np.cos((m+1)*Y/radius)
        fun3b = z*np.sin((m+1)*Y/radius)
        
        A[0,m+1] = (1/(2*length*circle*t)) * np.trapz(np.trapz(fun3a, y, axis = 0), x, axis = 0)
        B[0,m+1] = (1/(2*length*circle*t)) * np.trapz(np.trapz(fun3b, y, axis = 0), x, axis = 0)
    
    m = 0
    
    # A10-Al0 Ak0 and B10-Bk0
    for a in range(fc_x):
        fun4a = z*np.cos((a+1)*np.pi*X/length)
        
        A[a+1,0] = (1/(2*length*circle*t)) * np.trapz(np.trapz(fun4a, y, axis = 0), x, axis = 0)
      
    a = 0
    
    # Calculate zeta value
    for i in range(fc_x+1):
        for o in range(fc_y+1):
            zetakl = np.sqrt(A[i,o]**2 + B[i,o]**2)
        
        
    i = 0
    
    zeta = round(np.sum(zetakl), 4)
    
        
    np.savetxt("{}.FC_A".format(model),A)
    np.savetxt("{}.FC_B".format(model),B)
    
    return(z,A,B,zeta)

def Create_Node_File(model, length, na, nc, radius,datasavepath,number,no):
    myShellName = "{}".format(model)
    
    # File Names for the analysis 
    
    myFileName = str(myShellName)+"_Nodes.txt"
    myImageName = str(myShellName)+"_Shell.png"
    
    # Length of the cylinder
    
    myLength = length
    
    # Radius of the cylinder
    
    myRadius = radius
    
    # Wall thickness of the cylinder
    
    myThickness = 2 # Scale factor of imperfection
    
    # Number of Nodes in axial and circumferential direction
    
    na = na
    
    nc = nc
    
    
    # Displacement field
    
    w = np.zeros((na,nc))
    
    xyz = np.zeros((na*nc,3))
    
    # Fourier approach (1 - phase shift, 2 - cos , 3 - sin)
    
    case = 2
        
    ###############################################################################
    
    myFC_A_v = []
    myFC_B_v = []
    
    
    #Shells = ["Z07","Z08","Z10","Z11","Z12"]
    
    Shells = ["{}".format(model)]
    
    for i in Shells:
        A = np.loadtxt(str(i)+".FC_A")
        myFC_A_v.append(A)
        B = np.loadtxt(str(i)+".FC_B")
        myFC_B_v.append(B)
    
    
    n1 = len(A)
    n2 = len(A[0])
    
    n_imp = len(myFC_A_v)
    
    phim = np.zeros((n_imp,n1,n2))
    
    eps = np.zeros((n_imp,n1,n2))
    
    phim_list = np.zeros(n_imp)
    X_merge = []
    
    if case == 1:
        for j in range(0,n_imp,1):
            for x in range(0,n1,1):
                for y in range(0,n2,1):
                    #print(B_t[x][y])
                    if (myFC_A_v[j][x][y] >= 0):
                        phim[j][x][y] = np.arctan(myFC_B_v[j][x][y]/myFC_A_v[j][x][y])
                    elif (myFC_A_v[j][x][y] <= 0):
                        phim[j][x][y] = np.arctan(myFC_A_v[j][x][y]/myFC_B_v[j][x][y]) + np.pi
                    elif (myFC_A_v[j][x][y] == 0):
                        phim[j][x][y] = np.sign(myFC_B_v[j][x][y])*np.pi/2.0
                            
            #                
            for x in range(0,n1,1):
                for y in range(0,n2,1):
                    eps[j][x][y] = np.sqrt(myFC_A_v[j][x][y] * myFC_A_v[j][x][y] + myFC_B_v[j][x][y] * myFC_B_v[j][x][y]) 
            
            ###############################################################################
            
            phim_list = phim[j]
            phim_list = phim_list.ravel()
            phim_list = phim_list.tolist()
            eps_list = eps[j]
            eps_list = eps_list.ravel()
            eps_list = eps_list.tolist()
            X = phim_list + eps_list
            X = np.nan_to_num(X) 
            X_merge.append(X)
    elif case == 1 or case == 2:
        for j in range(0,n_imp,1):
            for x in range(0,n1,1):
                for y in range(0,n2,1):
                    phim[j][x][y] = myFC_A_v[j][x][y]
                    eps[j][x][y] = myFC_B_v[j][x][y] 
        
        ###############################################################################
        
            phim_list = phim[j]
            phim_list = phim_list.ravel()
            phim_list = phim_list.tolist()
            eps_list = eps[j]
            eps_list = eps_list.ravel()
            eps_list = eps_list.tolist()
            X = phim_list + eps_list
            X = np.nan_to_num(X) 
            X_merge.append(X)
                
        
        
        ###############################################################################
        
        # Calculate mean vector
        
        ###############################################################################
        
    
    r = len(X_merge[0])
    m = len(X_merge)
    mean_v =   np.zeros((r))  
    
    k = 0
    
    for i in range(0,r,1):
        for j in range(0,m,1):
            k = X_merge[j][i] + k
        mean_v[i] = k
        mean_v[i] = 1/m*mean_v[i]
        k = 0
        
        ###############################################################################
        
        # Calculate displacement field
        
        ###############################################################################
        
    
    phim_mean = np.zeros((n1,n2))
    
    eps_mean = np.zeros((n1,n2))
    
    
    phim_mean = mean_v[:len(mean_v)//2]
    phim_mean = np.reshape(phim_mean, (n1,n2))
    #phim_mean = np.transpose(phim_mean)
    eps_mean = mean_v[len(mean_v)//2:]
    eps_mean = np.reshape(eps_mean, (n1,n2))
    #eps_mean = np.transpose(eps_mean)
    
    
    
    for i in range(0,na,1):
        for j in range(0,nc,1):
            x = float(myLength)*(i)/(na)
            y = 2.0*np.pi*float(myRadius)*(j)/(nc)
            
            for k in range(0,n1,1):
                for l in range(0,n2,1):
                    w[i][j] = w[i][j] + np.cos((k)*np.pi*x/myLength)*(phim_mean[k][l]*np.cos((l)*y/myRadius) + eps_mean[k][l]*np.sin((l)*y/myRadius))
           
            w[i][j] = w[i][j] * myThickness
            xyz[(i)*nc+j][0] = (myRadius-w[i][j])*np.cos(y/myRadius)
            xyz[(i)*nc+j][1] = (myRadius-w[i][j])*np.sin(y/myRadius)
            xyz[(i)*nc+j][2] = x*1.016666671
    ##    
    ##################################################################################
    ##   
            
    # myFileName 
    # myImageName       
    
    nm = len(xyz)
    
    ABAQUS_NODES = np.zeros((nm,4))
    #
    for i in range(0,nm,1):
    
        ABAQUS_NODES[i][0] = i+1
        ABAQUS_NODES[i][1] = xyz[i][0]
        ABAQUS_NODES[i][2] = xyz[i][1]
        ABAQUS_NODES[i][3] = xyz[i][2]
        
    
    np.savetxt(myFileName,ABAQUS_NODES,fmt ='%i, %10.5f, %10.5f, %10.5f')
    
    cmap = cm.get_cmap('jet')
    p = plt.pcolor(w, cmap = cmap, vmin=-0.5, vmax=2.0)
    cb = plt.colorbar(p, orientation = 'vertical')
    plt.title('Initial imperfection of shell {}'.format(no), fontsize=14)
    plt.xlabel("Circumferential Nodes",fontsize=14)
    plt.ylabel("Axial Nodes",fontsize=14)
    #plt.xticks(np.arange(0,405,45))
    cb.set_label('Radial Displacement [mm]',fontweight='bold',fontsize=14)
    plt.savefig(r'D:\Desktop\Abaqus-python\CollectedData\Imp\Imperfection_{}.png'.format(no), dpi = 1000)
    plt.close()
    
    return(w)

def Create_Loading_Imperfection_Node_File(path, length, radius, angle, model, na, nc, number):
    f = np.genfromtxt(r"{}\DataSet_{}\{}_Nodes.txt".format(path, number, model),delimiter=',')
    
    a = angle / 180.0 * np.pi
    
    theta = np.transpose(np.linspace(0.0,3.14,120))
    
    for i in range(120):
        x = radius * np.cos(theta)
       
    h1 = radius * np.tan(a) + x * np.tan(a)
    h2 = np.flipud(h1)
    
    h = length - np.append(h1, h2)
    f[na*nc - nc:na*nc, 3] = h
    np.savetxt("{}.txt".format(model), f, fmt='%d, %10.5f, %10.5f, %10.5f') 

def Create_Inp_File(path, number, model):
    f = open('{}\DataSet_{}\{}.inp'.format(path, number, model),'w')
    f.write('''*HEADING
    IW1 - quasi-static Newton-Raphson
    **################################################################
    -------
    ** Geometry
    **--------------------------------
    *NODE,INPUT={}.txt
    **------------------------------------------------------- 
    *ELEMENT, TYPE=S4R, INPUT={}_element.txt
    **-------------------------------------------------------
    *ORIENTATION, SYSTEM=C, NAME=OID11
     0., 0., 0., 0., 0., 1.
     1, 90.
    **--------------------------------------------------------'''.format(model, model))
    f.close()

def Create_Element_File(na, nc, model):
    
    na = na
    nc = nc
    
    f = np.zeros((nc*(na-1),5))
    
    for i in range(nc*(na-1)):
        f[i,0:2] = i+1
        f[i,2] = i+2
        f[i,3] = nc+i+2
        f[i,4] = nc+i+1
        
    for n in range(na):
        f[nc*n-1,2] = nc*(n-1) + 1
        f[nc*n-1,3] = nc*n + 1
  
    np.savetxt("{}_element.txt".format(model), f, fmt='%d, %d, %d ,%d, %d')
    i = 0
    n = 0

def Import_Inp_File(path, number, model):
    mdb.ModelFromInputFile(name='{}'.format(model), inputFileName=r'{}\DataSet_{}\{}.inp'.format(path, number, model))

def Delete_Element(model):
    p = mdb.models[model].parts['PART-1']
    del p.features['OID11']
    a = mdb.models[model].rootAssembly
    a.regenerate()
    del a.features['PART-1-1']
    del a.features['Datum csys-1']

def Create_Part(platelength, part2, part3, model):
    #UpperPlate
    s1 = mdb.models[model].ConstrainedSketch(name='__profile__', sheetSize=200.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.rectangle(point1=(-1.0 * platelength / 2.0, -1.0 * platelength / 2.0), point2=(platelength / 2.0, platelength / 2.0))
    p = mdb.models[model].Part(name=part2, dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
    p = mdb.models[model].parts[part2]
    p.BaseShell(sketch=s1)
    s1.unsetPrimaryObject()
    p = mdb.models[model].parts[part2]
    del mdb.models[model].sketches['__profile__']
    
    #BottomPlate
    s = mdb.models[model].ConstrainedSketch(name='__profile__', sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.rectangle(point1=(-1.0 * platelength / 2.0, -1.0 * platelength / 2.0), point2=(platelength / 2.0, platelength / 2.0))
    p = mdb.models[model].Part(name=part3, dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
    p = mdb.models[model].parts[part3]
    p.BaseShell(sketch=s)
    s.unsetPrimaryObject()
    p = mdb.models[model].parts[part3]
    del mdb.models[model].sketches['__profile__']
    
    # Pick Ref Point
    p = mdb.models[model].parts[part2]
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    
    # Pick reference point for bottom plate
    p = mdb.models[model].parts[part3]
    p.ReferencePoint(point=(0.0, 0.0, 0.0))

def Create_Cylinder_Geometry_Set(model, setname, length, nc):
    p = mdb.models[model].parts['PART-1']
    e = p.elements
    elements = e[0:nc*(length-1)]
    p.Set(elements=elements, name=setname)

def Create_Material_Data_Input(model, material_name, density, e11, e22, e33, nu12, nu13, nu23, g12, g13, g23, x1t, x1c, x2t, x2c, s12, s13, lte, lce, tte, tce, damage_stabilization):
    mdb.models[model].Material(name=material_name)
    mdb.models[model].materials[material_name].Density(table=((density, ), ))
    mdb.models[model].materials[material_name].Elastic(type=ENGINEERING_CONSTANTS, table=((e11, e22, e33, nu12, nu13, nu23, g12, g13, g23), ))
    mdb.models[model].materials[material_name].HashinDamageInitiation(table=((x1t, x1c, x2t, x2c, s12, s13), ))
    mdb.models[model].materials[material_name].hashinDamageInitiation.DamageEvolution(type=ENERGY, table=((lte, lce, tte, tce), ))
    mdb.models[model].materials[material_name].hashinDamageInitiation.DamageStabilization(fiberTensileCoeff=damage_stabilization, fiberCompressiveCoeff=damage_stabilization, matrixTensileCoeff=damage_stabilization, matrixCompressiveCoeff=damage_stabilization)

def Assign_Material_Section(model, CylinderGeometrySet, angle, material, t):
    # Create Datum sys
    p = mdb.models[model].parts['PART-1']
    p.DatumCsysByThreePoints(name='Datum csys-1', coordSysType=CYLINDRICAL,origin=(0.0, 0.0, 0.0), line1=(1.0, 0.0, 0.0), line2=(0.0, 1.0, 0.0))
    
    # Assign fiber orientation
    p = mdb.models[model].parts['PART-1']
    region = p.sets[CylinderGeometrySet]
    orientation = mdb.models[model].parts['PART-1'].datums[4]
    mdb.models[model].parts['PART-1'].MaterialOrientation(region=region, orientationType=SYSTEM, axis=AXIS_2, localCsys=orientation, fieldName='', additionalRotationType=ROTATION_NONE, angle=0.0, additionalRotationField='')
    
    # Create section
    sectionLayer1 = SectionLayer(material=material, thickness=t, orientAngle=angle, numIntPts=3, plyName='')
    mdb.models[model].CompositeShellSection(name='Section-1', preIntegrate=OFF, idealization=NO_IDEALIZATION, symmetric=False, thicknessType=UNIFORM, poissonDefinition=DEFAULT, thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, integrationRule=SIMPSON, layup=(sectionLayer1, ))
    
    # Assign section
    p = mdb.models[model].parts['PART-1']
    region = p.sets[CylinderGeometrySet]
    p = mdb.models[model].parts['PART-1']
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

def Create_Assembly(model, part1, part2, part3, length):
    # Part1: Cylinder Part2: Upper Plate Part3: Bottom Plate
    a = mdb.models[model].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models[model].parts[part3]
    a.Instance(name='BottomPlate-1', part=p, dependent=ON)
    p = mdb.models[model].parts[part2]
    a.Instance(name='UpperPlate-1', part=p, dependent=ON)
    p = mdb.models[model].parts[part1]
    a1 = mdb.models[model].rootAssembly
    a1.Instance(name='PART-1-1', part=p, dependent=ON)
    a = mdb.models[model].rootAssembly
    a.translate(instanceList=('UpperPlate-1', ), vector=(0.0, 0.0, length))

def Create_Interaction(model):
    mdb.models[model].ContactProperty('IntProp-1')
    mdb.models[model].interactionProperties['IntProp-1'].TangentialBehavior(formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((0.3, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, fraction=0.005, elasticSlipStiffness=None)
    mdb.models[model].ContactExp(name='Int-1', createStepName='Initial')
    mdb.models[model].interactions['Int-1'].includedPairs.setValuesInStep(stepName='Initial', useAllstar=ON)
    mdb.models[model].interactions['Int-1'].contactPropertyAssignments.appendInStep(stepName='Initial', assignments=((GLOBAL, SELF, 'IntProp-1'), ))

def Create_Step(model, time_period, step_name):
    # mdb.models[model].ExplicitDynamicsStep(name=step_name, previous='Initial', timePeriod=time_period, timeIncrementationMethod=FIXED_USER_DEFINED_INC, userDefinedInc=time_period/100, improvedDtMethod=ON)
    mdb.models[model].ExplicitDynamicsStep(name=step_name, previous='Initial', timePeriod=time_period)

def Create_Condition_Set(model, TopCondSet, BotCondSet):
    a = mdb.models[model].rootAssembly
    r1 = a.instances['UpperPlate-1'].referencePoints
    refPoints1=(r1[2], )
    a.Set(referencePoints=refPoints1, name=TopCondSet)
    a = mdb.models[model].rootAssembly
    r1 = a.instances['BottomPlate-1'].referencePoints
    refPoints1=(r1[2], )
    a.Set(referencePoints=refPoints1, name=BotCondSet)

def Create_Condition(model,BotCondSet, TopCondSet, time_period, distance, step_name):
    a = mdb.models[model].rootAssembly
    region = a.sets[BotCondSet]
    mdb.models[model].EncastreBC(name='BC-1', createStepName='Initial', region=region, localCsys=None)
    a = mdb.models[model].rootAssembly
    region = a.sets[TopCondSet]
    mdb.models[model].DisplacementBC(name='BC-2', createStepName='Initial', region=region, u1=SET, u2=SET, u3=UNSET, ur1=SET, ur2=SET, ur3=SET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
    mdb.models[model].TabularAmplitude(name='Amp-1', timeSpan=STEP, smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (time_period, distance)))
    a = mdb.models[model].rootAssembly
    region = a.sets[TopCondSet]
    mdb.models[model].DisplacementBC(name='BC-3', createStepName=step_name, region=region, u1=UNSET, u2=UNSET, u3=-1.0, ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude='Amp-1', fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)

def Create_Mesh(model, bottompart, upperplate):
    p = mdb.models[model].parts[bottompart]
    p.seedPart(size=10.0, deviationFactor=0.1, minSizeFactor=0.1)
    p.generateMesh()
    p = mdb.models[model].parts[upperplate]
    p.seedPart(size=10.0, deviationFactor=0.1, minSizeFactor=0.1)
    p.generateMesh()

def Create_History_Output(model, monitoring_set, step_name, sampling_freq):
    regionDef=mdb.models[model].rootAssembly.allInstances['UpperPlate-1'].sets[monitoring_set]
    mdb.models[model].HistoryOutputRequest(name='Result', createStepName=step_name, variables=('U3', 'RF3'), frequency=sampling_freq, region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
def Create_Field_Output(model,step_name):
    mdb.models[model].FieldOutputRequest(name='F-Output-2', createStepName=step_name, variables=('S', 'U','RF'), timeInterval=1e-06)
    
def Create_Job(model):
    mdb.Job(name=model, model=model, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=5, activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=5)

def Submit_Job(model):
    mdb.jobs[model].submit(consistencyChecking=OFF)
    mdb.jobs[model].waitForCompletion()
    
    # time.sleep(200)
    
    # mdb.jobs[model].kill()
    # print('Time over!')
    # mdb.jobs[model].kill()
    # time.sleep(1)
    # mdb.jobs[model].kill()

def Open_ODB_and_Write_NodeSet_data_to_text(model,step_name,variable_name,set_name,Variable_component):
    # open ODB file - ABAQUS Result file
    odb = session.openOdb(str(model)+'.odb')
    
    # list for the VARIABLE you want to evaluate
    Variable_v = []
    
    # analysis step for your VARIABLE
    lastStep=odb.steps[step_name]
    
    #loop over all increments of the analysis step and save VARIABLE information from each increment
    for x in range(len(lastStep.frames)):
        lastFrame = lastStep.frames[x]
        Variable = lastFrame.fieldOutputs[variable_name]
        center = odb.rootAssembly.nodeSets[set_name]
        centerRForce = Variable.getSubset(region=center)
       
        # loop over the VARIABLE and save component (x,y,z - 0,1,2) to list
        for i in centerRForce.values:
            Variable_vr = [i.data[Variable_component]* -1]
            Variable_v = Variable_v + Variable_vr
            
    # write VARIABLE - component to text file
    
    np.savetxt(str(variable_name)+'_'+str(model)+'.txt',Variable_v)

def Open_ODB_and_Write_Max_Value_of_NodeSet_data_to_text(model,step_name,variable_name,variable_name_max,set_name,Variable_component):
    # open ODB file - ABAQUS Result file
    odb = session.openOdb(str(model)+'.odb')
    
    # list for the VARIABLE you want to evaluate
    Variable_v = []
    
    # analysis step for your VARIABLE
    lastStep=odb.steps[step_name]
    
    #loop over all increments of the analysis step and save VARIABLE information from each increment
    for x in range(len(lastStep.frames)):
        lastFrame = lastStep.frames[x]
        Variable = lastFrame.fieldOutputs[variable_name]
        center = odb.rootAssembly.nodeSets[set_name]
        centerRForce = Variable.getSubset(region=center)
       
        # loop over the VARIABLE and save component (x,y,z - 0,1,2) to list
        for i in centerRForce.values:
            Variable_vr = [i.data[Variable_component] * -1]
            Variable_v = Variable_v + Variable_vr 
    
    # Max value of Variable_v
    Max_Variable = np.max(Variable_v)
    
    index = Variable_v.index(Max_Variable)
    
    Max_Variable_v = Max_Variable
    return(round(Max_Variable_v,2),index)
    # write VARIABLE - component to text file
    
    # np.savez(str(variable_name_max)+'_'+str(model),a = Max_Variable_v, b = radius, c = length, d = thickness, e = BPangle, f = zeta)

#-----------------------------------------------------------------------------------------------------

def Plot_Load_Displacement_plot(savepath,number,model,no):
    U = np.genfromtxt(r'{}\DataSet_{}\U_{}.txt'.format(savepath,number,model))
    RF = np.genfromtxt(r'{}\DataSet_{}\RF_{}.txt'.format(savepath,number,model))
        
    plt.plot(U, RF)
    plt.axis([0,3,0,1000000])
    plt.title('L-D plot of shell {}'.format(no), fontsize=14)
    plt.savefig(r'D:\Desktop\Abaqus-python\CollectedData\Plot\L-D plot_{}.png'.format(no), dpi = 1000)
    plt.close()

def Open_ODB_and_Extract_Buckling_Mode(model,savepath,number,bucklingmode,na,nc, frameno):  
    o3 = session.openOdb(name='{}/DataSet_{}/{}.odb'.format(savepath,number,model))
    session.viewports['Viewport: 1'].setValues(displayedObject=o3)
    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT,'Magnitude'), )
    session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=CONTOURS_ON_DEF)
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=frameno)
    session.Path(name='Shell', type=NODE_LIST, expression=(('PART-1-1', ('1:{}'.format(na*nc), )), ))
    pth = session.paths['Shell']
    x0 = session.XYDataFromPath(name='BucklingMode', path=pth, includeIntersections=False, projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, projectionTolerance=0, shape=DEFORMED, labelType=SEQ_ID)
    session.writeXYReport(fileName='{}\DataSet_{}\{}.txt'.format(savepath,number,bucklingmode), xyData=(x0,))  
    
    with open(r'{}\DataSet_{}\{}.txt'.format(savepath,number,bucklingmode),'r') as fin:
        data = fin.read().splitlines(True)
    
    with open(r'{}\DataSet_{}\{}.txt'.format(savepath,number,bucklingmode),'w') as fout:
        fout.writelines(data[3:])
    f1 = np.genfromtxt(r'{}\DataSet_{}\{}.txt'.format(savepath,number,bucklingmode))
    
    b = np.reshape(f1[:,1],(na,nc))
    
    x = np.linspace(1,nc,nc)
    y = np.linspace(1,na,na)
    
    cmap = cm.get_cmap('jet')
    p = plt.pcolor(b,cmap = cmap, vmin=-0.5, vmax=2.0)
    cb = plt.colorbar(p, orientation = 'vertical')
    plt.xticks(np.arange(0, len(x), 20))
    plt.yticks(np.arange(0, len(y), 5))
    plt.title(bucklingmode, fontsize=14)
    plt.xlabel('Circumferential nodes', fontsize=14)
    plt.ylabel('Axial nodes', fontsize=14)
    cb.set_label('Displacement Magnitude [mm]',fontweight='bold',fontsize=14)
    plt.savefig(r'D:\Desktop\Abaqus-python\CollectedData\BM\{}.png'.format(bucklingmode), dpi = 1000)
    plt.close()
    
    return(b)
    