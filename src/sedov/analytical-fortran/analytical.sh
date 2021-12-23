rm -f *.png 

echo "load \"analytical.gnu\"" | gnuplot

eog "density.png",  \
    "energy.png",   \
    "pressure.png", \
    "velocity.png"  &
