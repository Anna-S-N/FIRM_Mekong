#!/bin/sh

python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'new_modelled_baseline' -b 'batteries'
python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'modelled_newbuild' -b 'batteries'
python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'new_modelled_baseline' -b 'noBatteries'
python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'modelled_newbuild' -b 'noBatteries'

