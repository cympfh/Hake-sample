# invoke Hake
run:
	hake run \
		--max acc \
		EPOCH=10..30 \
		BS=16,32,64 \
		HD=5..50 \
		ACT=sigmoid,relu \
		OPT=sgd,adam

clean:
	rm -rf .hake/ models/*.h5
