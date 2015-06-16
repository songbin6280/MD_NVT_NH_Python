# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 00:19:52 2014

@author: Bin Song
To use the code: python code -f trajectory
or python -c volume number/volum

The trajecotry file has the same format of a trajectory file ouput by LAMMPS
Provide an input file with the same format in the sample.

"""
import numpy as np
import sys
import math
import constants as cst
import argparse
#function to read file
def read_file(filename):
    try:
        f=open(filename,"r")
        lines=f.readlines()
        f.close
        return lines
    except IOError:
        print("The file {} cannot be opened.".format(filename))
        sys.exit()
#output file
def output(filename):
    try:
        f=open(filename, "w")
        return f
    except IOError:
        print("Output file {} cannot be opened.".format(filename))
        sys.exit()
#assign initial velocities to the system according to T        
def init(nparticles, T, mass): 
    vel = np.random.uniform(-0.5, 0.5, nparticles*3).reshape(nparticles,3) #assign velocity using uniform distribution
    cvmx=vel[:,0].sum()/nparticles
    cvmy=vel[:,1].sum()/nparticles
    cvmz=vel[:,2].sum()/nparticles
    for i in range(nparticles):    # make sure the overall momentum is zero
        vel[i,0] -= cvmx
        vel[i,1] -= cvmy
        vel[i,2] -= cvmz
    T_temp = np.square(vel).sum()*mass*(10**7)/3.0/cst.Rg/nparticles #code below to ensure velocity scales with T
#print T_temp
    scale = math.sqrt(T/T_temp)
    vel=np.multiply(scale, vel)
    return vel        
#create a box and assign positions of atoms if necessary
#The box will be divided into cubic lattices with one atom in each lattice.
def create(bx, by, bz, nparticles, n_3):
    ix, iy, iz = [0, 0 ,0]
    coord=np.zeros(shape=[nparticles,3], dtype="float")
    for i in range(nparticles):
        coord[i,0]=float((ix+0.5)*bx/n_3)
        coord[i,1]=float((iy+0.5)*by/n_3)
        coord[i,2]=float((iz+0.5)*bz/n_3)
        ix += 1
        if ix == n_3:
            ix = 0
            iy += 1
        if iy == n_3:
            iy = 0
            iz +=1
    return coord
# calculate potential energy and forces using Lennard-Jones 12-6 potetial
def pe_force(sigma12, sigma6, epsilon, rc2, bx, by, bz, nparticles, coord):
    force=np.zeros(shape=[nparticles,3], dtype="float")
    pe=0.0
    vir = 0.0
    for i in range(nparticles-1):
        for j in range(i+1,nparticles):
            rx=coord[i,0]-coord[j,0]
            ry=coord[i,1]-coord[j,1]
            rz=coord[i,2]-coord[j,2]
            rx=rx-bx*int(math.floor(rx/bx+0.5))
            ry=ry-by*int(math.floor(ry/by+0.5))
            rz=rz-bz*int(math.floor(rz/bz+0.5))
            r2 = rx**2 + ry**2 + rz**2
            if r2 < rc2:
                r6i = 1.0 /r2**3
                pe+=4*epsilon*(sigma12*r6i*r6i-sigma6*r6i)
                f = 48*epsilon*(sigma12*r6i*r6i-0.5*sigma6*r6i)
                force[i,0]+=rx*f/r2
                force[j,0]-=rx*f/r2
                force[i,1]+=ry*f/r2
                force[j,1]-=ry*f/r2
                force[i,2]+=rz*f/r2
                force[j,2]-=rz*f/r2
                vir += f
    return pe, force, vir
# nose-hoover chain. Follow the algorithm 31 from Frenkel & Smit's book:
# Understanding Molecular simulaton From Algorithms to Applications.
def nhchain(Q, dt, dt_2, dt_4, dt_8, nparticles, vxi, xi, ke, vel):
    G2 = (Q[0]*vxi[0]*vxi[0]-T*cst.kb)
    vxi[1] += G2*dt_4
    vxi[0] *= math.exp(-vxi[1]*dt_8)
    G1 = (2*ke*10**7/cst.NA-3*nparticles*T*cst.kb)/Q[0]
    vxi[0] += G1*dt_4
    vxi[0] *= math.exp(-vxi[1]*dt_8)
    xi[0]+=vxi[0]*dt_2
    xi[1]+=vxi[1]*dt_2
    s=math.exp(-vxi[0]*dt_2)
    vel=np.multiply(s,vel)
    ke*=(s*s)
    vxi[0]*=math.exp(-vxi[1]*dt_8)
    G1=(2*ke*10**7/cst.NA-3*nparticles*T*cst.kb)/Q[0]
    vxi[0]+=G1*dt_4
    vxi[0]*=math.exp(-vxi[1]*dt_8)
    G2=(Q[0]*vxi[0]*vxi[0]-T*cst.kb)/Q[1]
    vxi[1]+=G2*dt_4
    return ke

# Start of the main program. Reading of -f signals a data file. Reading of -c asks the program to create a box
parser =argparse.ArgumentParser(description='Tell the code to read in a data file or create a data file')
parser.add_argument('-f', action="store", dest="data")
parser.add_argument('-c', action="store", nargs=2, dest="c")

args=parser.parse_args()
if args.data is None and args.c is None:
    print("Invalid input: No data file is indicated, and no box creation parameters are availabel")
    sys.exit()
if args.data is not None:
    datafile=read_file(args.data)
    for i, item in enumerate(datafile):
        if i < 9:
            if i == 0:
                continue
            elif i== 1:
                continue
            elif i==2:
                continue
            elif i==3:
                nparticles=int(item.strip())
                coord=np.empty(shape=[nparticles,3], dtype="float")
            elif i==4:
                continue
            elif i==5:
                x = item.strip().split(" ")
                xlo = float(x[0])
                xhi = float(x[1])
	        bx= xhi-xlo
            elif i==6:
                y= item.strip().split(" ")
                ylo = float(y[0])
                yhi = float(y[1])
	        by = yhi - ylo
            elif i==7:
                z = item.strip().split(" ")
                zlo = float(z[0])
                zhi = float(z[1])
	        bz = zhi-zlo
            elif i==8:
                continue
        else:
            line = item.strip().split(" ")
            coord[i-9,0]=line[2]
            coord[i-9,1]=line[3]
            coord[i-9,2]=line[4]
    else:
        if i != 8+nparticles:
            print("Input data file has a wrong number of atoms")
            sys.exit()
    V = bx * by * bz
    print("Data file read")
if args.c is not None:
    V=float(args.c[0])
    rho=float(args.c[1])
    bx=by=bz=V**(1.0/3)
    xlo=ylo=zlo=0.0
    xhi=yhi=zhi=bx
    nparticles=int(V*rho)
    n_3=int(math.floor(nparticles**(1.0/3)+0.5))
    nparticles=n_3**3
    print nparticles
    coord=create(bx, by, bz, nparticles, n_3)
    print("Box creation completed")
#read the input file containing lennard-jones parameters, mass of an atom, duration of a time step and number of steps.  
para=read_file("input.txt")
lines=[block.strip() for block in para if block[0] is not "#"]
lj= lines[0].split(" ")
sigma=float(lj[0])
epsilon=float(lj[1])
rc=float(lj[2])
dt=float(lines[1])
nsteps=int(lines[2])
mass=float(lines[3])
T=float(lines[4])
thermo=int(lines[5]) #frequency to output the log file
dump=int(lines[6]) # frequency to write trajectory file
print("Input file read")
print ("sigma epsilon rc dt nsteps mass T thermo dump")
print sigma, epsilon, rc, dt, nsteps, mass, T, thermo, dump 

trj=output("nvt.lammpstrj")
print("Trajectory file has been created")
log=output("log.dat")
print("Log file has been created")
log.write("timestep potential_eng(KJ/mol) kinetic_eng(KJ/mol) T(K) Press(atm)\n")
#compute parameters
rc2=rc**2
#print rc2
dt_2=0.5*dt
dt_4=0.5*dt_2
dt_8=0.5*dt_4
sigma6=sigma**6
sigma12=sigma6**2

Q=[1000.0, 1000.0]
xi=[0.0, 0.0]
vxi=[0.0, 0.0]

vel=init(nparticles, T, mass)
#initialize velocities
print("Velocity initialization completed.")

ke=0.5*mass*np.square(vel).sum()

#main MD code
for i in range(nsteps):
    print("Working on step {}.".format(i))
    #algorithm 32 in Frenkel & Smit's book 
    ke= nhchain(Q, dt, dt_2, dt_4, dt_8, nparticles, vxi, xi, ke, vel) #update the Nose-Hoover chain
    for j in range(nparticles):
        coord[j,0]+=vel[j,0]*dt_2
        coord[j,1]+=vel[j,1]*dt_2
        coord[j,2]+=vel[j,2]*dt_2
        if coord[j,0] < xlo: 
            coord[j,0]+=bx
        if coord[j,0] > xhi:
            coord[j,0]-=bx
        if coord[j,1] < ylo:
            coord[j,1]+=by
        if coord[j,1] > yhi:
            coord[j,1]-=by
        if coord[j,2] < zlo:
            coord[j,2]+=bz
        if coord[j,2] > zhi:
            coord[j,2]-=bz
    pe, force, vir=pe_force(sigma12, sigma6, epsilon, rc2, bx, by, bz, nparticles, coord)
    
    for j in range(nparticles):
        vel[j,0]+=dt*force[j,0]/mass*0.0001  #0.0001 
        vel[j,1]+=dt*force[j,1]/mass*0.0001
        vel[j,2]+=dt*force[j,2]/mass*0.0001
        coord[j,0]+=vel[j,0]*dt_2
        coord[j,1]+=vel[j,1]*dt_2
        coord[j,2]+=vel[j,2]*dt_2
        if coord[j,0] < xlo: 
            coord[j,0]+=bx
        if coord[j,0] > xhi:
            coord[j,0]-=bx
        if coord[j,1] < ylo:
            coord[j,1]+=by
        if coord[j,1] > yhi:
            coord[j,1]-=by
        if coord[j,2] < zlo:
            coord[j,2]+=bz
        if coord[j,2] > zhi:
            coord[j,2]-=bz
    ke=0.5*mass*np.square(vel).sum()
    ke= nhchain(Q, dt, dt_2, dt_4, dt_8, nparticles, vxi, xi, ke, vel)
    if i%dump == 0:
          trj.write("ITEM: TIMESTEP\n")
          trj.write(str(i)+"\n")
          trj.write("ITEM: NUMBER OF ATOMS\n")
          trj.write(str(nparticles)+"\n")
          trj.write("ITEM: BOX BOUNDS\n")
          trj.write(str(xlo)+" "+str(xhi)+"\n")
          trj.write(str(ylo)+" "+str(yhi)+"\n")
          trj.write(str(zlo)+" "+str(zhi)+"\n")
          trj.write("ITEM: ATOMS id type x y z\n")
          for j in range(nparticles):
              trj.write(str(j+1)+" 1 "+str(coord[j,0])+" "+str(coord[j,1])+" "+str(coord[j,2])+"\n")
          trj.flush()
    if i%thermo == 0:
          log.write(str(i)+" "+str(pe)+" "+str(ke*10000)+" "+str(ke*2*(10**7)/3.0/nparticles/cst.Rg)+" "+str((ke*2.0/3.0/V*10**4+vir/3.0/V)/cst.NA*10**28/1.013)+"\n")
          log.flush()
print("Simulation Completed")
trj.close()
log.close()
    


                

    


    
    
    
    

    
            
