mutation=(0.3 0.2 0.1 0.05 0.01)
retain=(0.15 0.2 0.25 0.3)
diversity=0.05
ite=2000
pop_count=40
parent=(random roullette)
parentN=(prandom proullette)
survivor=(truncated roullette)
survivorN=(strunc sroullette)
rules=15


for i in {0..4}
	do for j in {0..3}
		do for k in {0..0}
			do for l in {0..0}
				do	pypy gabil_optimizer.py 	--action train \
												--input credit-screening/crx.data \
												--count_rules $rules \
												--iterations $ite \
												--pop_count $pop_count \
												--mutate_prob ${mutation[$i]} \
												--retain ${retain[$j]} \
												--parents ${parent[$k]} \
												--survivors ${survivor[$l]} \
												--output $rules-${parentN[$k]}-${survivorN[$l]}-$ite-$pop_count-${mutation[$i]}-${retain[$j]}-$diversity #>> Log.txt
					echo "Finished $rules-${parentN[$k]}-${survivorN[$l]}-$ite-$pop_count-${mutation[$i]}-${retain[$j]}-$diversity"
			done
		done
	done
done
