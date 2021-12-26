#  SITUACION DE LAS VARIABLES EN EL FICHERO DE DATOS
####################################################
#
#  1: radio al centro de masas (r) 
#  2: Density (rho)
#  3: Internal Energy in log10 (log10(U))
#  4: Pressure (P)
#  5: velocidad (|v|)
#  6: Sound Velocity (cs)
#  7: Density normalized (rho/rho0)
#  8: Density shock normalized (rho/rho_Shock)
#  9: Pressure shock normalized (P/P_Shock)
# 10: Velocity shock normalized (|v|/V_Shock)
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
 set ylabel "{/Symbol r} (g.cm^{-2}) [normalized]"
 unset title
 set key left top
 set output 'density.png'
 #set xrange [0:25]
 #set yrange[0:4]

 plot solSedov  u 1:7    w l ls 3 title "{/Symbol r}_{norm}"

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

 plot solSedov  u 1:3    w l ls 3 title "log10(U)"

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

 plot solSedov  u 1:4    w l ls 3 title "P"

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

 plot solSedov  u 1:5    w l ls 3 title "|V|"

 set autoscale x
 set autoscale y
 unset format 

######################################################################################################

set xlabel "R (cm)"
set ylabel  "{/Symbol r} (g.cm^{-2}) [normalized]"
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
plot solSedov  u 1:7    w l ls 1 title "{/Symbol r}_{norm}" axes x1y1, \
     solSedov  u 1:4    w l ls 3 title "Pressure" axes x1y2

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
set ylabel  "{/Symbol r} (g.cm^{-2}) [normalized]"
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

plot solSedov  u 1:7    w l ls 1 title "{/Symbol r}_{norm}" axes x1y1, \
     solSedov  u 1:5    w l ls 3 title "Velocity" axes x1y2

set autoscale x
set autoscale y
set autoscale y2
set ytics mirror
set y2tics mirror
unset y2tics
set y2label ""
unset format 

######################################################################################################
