#  SITUACION DE LAS VARIABLES EN EL FICHERO DE DATOS
####################################################
#
#  1: nStep     
#  2: radio al centro de masas (r) 
#  3: Density (rho)
#  4: Energy (U)
#  5: Pressure (P)
#  6: velocidad (|v|)
#  7: Sound Velocity (cs)
#
####################################################

set size ratio 1

set autoscale x
set autoscale y

set term pngcairo dashed enhanced font 'Verdana,14'


# color definitions
set style line 1 lc rgb '#ff0000' pt 2  ps 1 lt 1 lw 2 # --- red
set style line 2 lc rgb '#00ff00' pt 5  ps 1 lt 1 lw 2 # --- green
set style line 3 lc rgb '#0000ff' pt 7  ps 1 lt 1 lw 2 # --- blue
set style line 4 lc rgb '#ffff00' pt 9  ps 1 lt 1 lw 2 # --- yellow
set style line 5 lc rgb '#ff00ff' pt 11 ps 1 lt 1 lw 2 # --- magenta
set style line 6 lc rgb '#00ffff' pt 13 ps 1 lt 1 lw 2 # --- cyan
set style line 7 lc rgb '#000000' pt 2  ps 1 lt 1 lw 2 # --- black

set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12


solSedov="theoretical.dat"

######################################################################################################

 set xlabel "R (cm)"
 set ylabel "{/Symbol r} (g.cm^{-2})"
 unset title
 set key left top
 set output 'density.png'
 #set xrange [0:25]
 #set yrange[0:4]

 plot solSedov  u 2:3    w l ls 1 title "{/Symbol r} Sim   ", \
      solSedov  u 2:3    w l ls 3 title "{/Symbol r} Teoric"

 set autoscale x
 set autoscale y
 unset format 

######################################################################################################

 set xlabel "R (cm)"
 set ylabel "log10(U) (erg.g^{-1})"
 unset title
 set key left top
 #set logscale y
 #set format y '%.1e'
 set output 'energy.png'
 #set xrange [0:25]
 #set yrange[0:4]

 plot solSedov  u 2:(log10($4))    w l ls 1 title "U Sim   ", \
      solSedov  u 2:(log10($4))    w l ls 3 title "U Teoric"

 #unset logscale y
 set autoscale x
 set autoscale y
 unset format 

######################################################################################################

 set xlabel "R (cm)"
 set ylabel "P (erg.cm^{-2})"
 unset title
 set key left top
 set output 'pressure.png'
 #set xrange [0:25]
 #set yrange[0:4]

 plot solSedov  u 2:5    w l ls 1 title "P Sim   ", \
      solSedov  u 2:5    w l ls 3 title "P Teoric"

 set autoscale x
 set autoscale y
 unset format 

######################################################################################################

 set xlabel "R (cm)"
 set ylabel "|V| (cm.s^{-1})"
 unset title
 set key left top
 set output 'velocity.png'
 #set xrange [0:25]
 #set yrange[0:4]

 plot solSedov  u 2:6    w l ls 1 title "|V| Sim   ", \
      solSedov  u 2:6    w l ls 3 title "|V| Teoric"

 set autoscale x
 set autoscale y
 unset format 

######################################################################################################

set xlabel "R (cm)"
set ylabel  "{/Symbol r} (g.cm^{-2})"
set y2label "P (dyn.cm)"
#set format y '%.1e'
#set format y2 '%.1e'
unset title
set key left top
set output 'rhoPressure.png'
set y2tics 
set y2tics nomirror
set ytics nomirror
#set xrange [0:25]

#set yrange[0:4]
plot solSedov  u 2:3    w l ls 1 title "{/Symbol r}" axes x1y1, \
     solSedov  u 2:5    w l ls 3 title "Pressure" axes x1y2

set autoscale x
set autoscale y
set autoscale y2
set ytics mirror
set y2tics mirror
unset y2tics
set y2label ""
unset format 

######################################################################################################

set xlabel "R (cm)"
set ylabel  "{/Symbol r} (g.cm^{-2})"
set y2label "|V| (cm~s^{-1})"
#set format y '%.1e'
#set format y2 '%.1e'
unset title
set key left top
set output 'rhoVelocity.png'
set y2tics 
set y2tics nomirror
set ytics nomirror
#set xrange [0:25]
#set yrange[0:4]

plot solSedov  u 2:3    w l ls 1 title "{/Symbol r}" axes x1y1, \
     solSedov  u 2:6    w l ls 3 title "Velocity" axes x1y2

set autoscale x
set autoscale y
set autoscale y2
set ytics mirror
set y2tics mirror
unset y2tics
set y2label ""
unset format 

######################################################################################################
