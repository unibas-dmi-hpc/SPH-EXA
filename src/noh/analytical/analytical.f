	program analytical
	implicit none

c..tests the noh solver

c..declare
	character*80 outfile,string
	integer i,nstep,iargc
	double precision  time,zpos,
     1			rho0,vel0,gamma,xgeom,
     2			den,ener,pres,vel,
     3			zlo,zhi,zstep,value

c..popular formats
01	format(1x,t4,a,t8,a,t22,a,t36,a,t50,a,t64,a,t78,a,t92,a)
02	format(1x,i4,1p8e12.4)

c..input parameters in cgs
	time = 0.3d0
	rho0 = 1.0d0
	vel0 = -1.0d0
	gamma = 5.0d0/3.0d0
	xgeom = 3.0d0

c..number of grid points, spatial domain, spatial step size
	nstep = 100
	zlo = 0.0d0
	zhi = 1.0d0
	zstep = (zhi - zlo)/float(nstep)

c..output file
	outfile = 'theoretical.dat'
	open(unit=2,file=outfile,status='unknown')
	write(2,02) nstep,time
	write(2,01) 'i','x','rho','e','p','v'

c..to match hydrocode output, use the mid-cell points
	do i=1,nstep
	 zpos = zlo + 0.5d0*zstep + float(i-1)*zstep
	 call noh_1d(time,zpos,
     1		rho0,vel0,gamma,xgeom,
     2		den,ener,pres,vel)


	 write(2,40) i,zpos,den,ener,pres,vel
40	 format(1x,i4,1p8e14.6)

	enddo

c..close up stop
	close(unit=2)
	end


	subroutine noh_1d(time,xpos,
     1 			rho1,u1,gamma,xgeom,
     2			den,ener,pres,vel)
	implicit none
	save

c..solves the standard case, (as opposed to the singular or vacuum case),
c..constant density (omega = 0) sedov problem in one-dimension.

c..input:
c..	time = temporal point where solution is desired seconds
c..	xpos = spatial point where solution is desired cm

c..output:
c..	den = density g/cm**3
c..	ener = specific internal energy erg/g
c..	pres = presssure erg/cm**3
c..	vel = velocity cm/sh

c..declare the pass
	double precision time,xpos,
     1			rho1,u1,gamma,xgeom,
     3			den,ener,pres,vel

c..local variables
	double precision gamm1,gamp1,gpogm,xgm1,us,r2,rhop,rho2,u2,e2,p2

c..some parameters
	gamm1 = gamma - 1.0d0
	gamp1 = gamma + 1.0d0
	gpogm = gamp1 / gamm1
	xgm1 = xgeom - 1.0d0

c..immediate post-chock values using strong shock relations
c..shock velocity, position, pre- and post-shock density,
c..flow velocity, internal energy, and pressure

	us = 0.5d0 * gamm1 * abs(u1)
	r2 = us * time
	rhop = rho1 * (1.0d0 - (u1*time/r2))**xgm1
	rho2 = rho1 * gpogm**xgeom
	u2 = 0.0d0
	e2 = 0.5d0 * u1**2
	p2 = gamm1 * rho2 * e2

c..if we are farther out than the shock front
	if (xpos .gt. r2) then
		den = rho1 * (1.0d0 - (u1*time/xpos))**xgm1
		vel = u1
		ener = 0.0d0
		pres = 0.0d0

c..if we are between the origin and the shock front
	else
		den = rho2
		vel = u2
		ener = e2
		pres = p2
	end if

	return
	end

