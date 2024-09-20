#!/bin/sh

python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'modelled_baseline' -b 'batteries'
python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'modelled_newbuild' -b 'batteries'
python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'modelled_baseline' -b 'noBatteries'
python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'modelled_newbuild' -b 'noBatteries'

python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'flexible' -b 'batteries'
python3 Optimisation.py -i 1000 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'flexible' -b 'noBatteries'

python3 Optimisation.py -i 1000 -p 8 -e 99 -n 'KH' -s 'no_nuclear' -f 'flexible' -b 'batteries'
python3 Optimisation.py -i 1000 -p 8 -e 99 -n 'TH' -s 'no_nuclear' -f 'flexible' -b 'batteries'
python3 Optimisation.py -i 1000 -p 8 -e 99 -n 'Laos_Iso_Grid' -s 'no_nuclear' -f 'flexible' -b 'batteries'
python3 Optimisation.py -i 1000 -p 8 -e 99 -n 'Vietnam_Iso_Grid' -s 'no_nuclear' -f 'flexible' -b 'batteries'
python3 Optimisation.py -i 1000 -p 8 -e 99 -n 'KH' -s 'no_nuclear' -f 'flexible' -b 'noBatteries'
python3 Optimisation.py -i 1000 -p 8 -e 99 -n 'TH' -s 'no_nuclear' -f 'flexible' -b 'noBatteries'
python3 Optimisation.py -i 1000 -p 8 -e 99 -n 'Laos_Iso_Grid' -s 'no_nuclear' -f 'flexible' -b 'noBatteries'
python3 Optimisation.py -i 1000 -p 8 -e 99 -n 'Vietnam_Iso_Grid' -s 'no_nuclear' -f 'flexible' -b 'noBatteries'

python3 Optimisation.py -i 1000 -p 8 -e 3 -n 'Mekong_Grid' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 3 -n 'KH' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 3 -n 'TH' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 3 -n 'Laos_Iso_Grid' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 3 -n 'Vietnam_Iso_Grid' -s 'no_nuclear' -f 'flexible'

python3 Optimisation.py -i 1000 -p 8 -e 10 -n 'Mekong_Grid' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 10 -n 'KH' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 10 -n 'TH' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 10 -n 'Laos_Iso_Grid' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 10 -n 'Vietnam_Iso_Grid' -s 'no_nuclear' -f 'flexible'

python3 Optimisation.py -i 1000 -p 8 -e 20 -n 'Mekong_Grid' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 20 -n 'KH' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 20 -n 'TH' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 20 -n 'Laos_Iso_Grid' -s 'no_nuclear' -f 'flexible'
python3 Optimisation.py -i 1000 -p 8 -e 20 -n 'Vietnam_Iso_Grid' -s 'no_nuclear' -f 'flexible'