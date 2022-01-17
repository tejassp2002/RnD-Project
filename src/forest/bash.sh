for x in 1 2 3 4 5 6 7 8 9 10
do
	echo "$x"
	python3 -m newtrain --iter $x
done
export PATH=/usr/bin/python3:$PATH