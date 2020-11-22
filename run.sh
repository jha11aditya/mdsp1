
for fil in input/*.png
do
convert $fil -compress none $fil'.ppm'
done



python3 main.py $1

rm input/*.ppm
