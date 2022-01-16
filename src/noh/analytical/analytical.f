      program analytical
      implicit none

      character*80     outfile,string
      integer          i,nstep,nmax
      parameter        (nmax = 1000)
      real*16          time,zpos(nmax),
     1                 rho0,vel0,gamma,xgeom,
     2                 den,ener,pres,vel,
     3                 zlo,zhi,zstep

      ! popular formats
 01   format(15x,a,13x,a,15x,a,15x,a,13x,a,14x,a,8x,a,8x,a)
 02   format(10(2x,1pe14.6))

      nstep   = nmax                     !
      xgeom   = 3.0q0				     ! geometry type (dimensions)
      outfile = 'theoretical.dat'        ! output file name

      ! input parameters in cgs
      time   =  1.5q0                    ! Time where the solution is calculated
      rho0   =  1.0q0				     ! Rho inicial
      vel0   = -1.0q0                    !
      gamma  =  5.0q0/3.0q0              !

      ! Domininio espacial (0-1.0)
      zlo = 0.0q0
      zhi = 1.0q0
      zstep = (zhi - zlo)/float(nstep)
      do i=1,nstep
       	zpos(i)   = zlo + 0.5q0*zstep + float(i-1)*zstep
      enddo
      
      ! output file
      open(unit=2,file=outfile,status='unknown')
      write(2,01) 'r','rho','u','p','vel'

      ! to match hydrocode output, use the mid-cell points
      do i=1,nstep
      
         call noh_1d(time,zpos(i),
     1               xgeom,
     2               rho0,vel0,gamma,
     3               den,ener,pres,vel)

         write(2,02) zpos(i),
     1               den,
     2               ener,
     3               pres,
     4               vel
     
      enddo
      close(unit=2)
      
      write(6,*)
      
      end


      subroutine noh_1d(time,xpos,
     1                  xgeom_in,
     2                  rho0,vel0,gam0,
     3                  den,ener,pres,vel)
      implicit none

!   solves the standard case, (as opposed to the singular or vacuum case),
!   constant density (omega = 0) sedov problem in one-dimension.

!   input:
!       time = temporal point where solution is desired seconds
!       xpos = spatial point where solution is desired cm

!   output:
!       den  = density g/cm**3
!       ener = specific internal energy erg/g
!       pres = presssure erg/cm**3
!       vel  = material speed cm/s


!   declare the pass
      real*16          time,xpos,
     1                 xgeom_in,
     2                 rho0,vel0,gam0,
     3                 den,ener,pres,vel

!   local variables
      real*16          gamm1,gamp1,gpogm,xgm1, r2,rhop

!   some parameters
      gamm1  = gam0 - 1.0q0
      gamp1  = gam0 + 1.0q0
      gpogm  = gamp1 / gamm1
      xgm1   = xgeom_in - 1.0q0

!   immediate post-chock values using strong shock relations
      r2   = 0.5q0 * gamm1 * abs(vel0) * time
      rhop = rho0 * (1.0q0 - (vel0*time/r2))**xgm1

      if (xpos .gt. r2) then
      
            ! if we are farther out than the shock front
            den  = rho0 * (1.0q0 - (vel0*time/xpos))**xgm1
            ener = 0.0q0
            pres = 0.0q0
            vel  = vel0
      else
            ! if we are between the origin and the shock front
            den  = rho0  * gpogm**xgeom_in
            ener = 0.5q0 * vel0**2
            pres = gamm1 * den * ener
            vel  = 0.0q0
      end if

      return
      end
