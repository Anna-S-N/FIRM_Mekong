#!/bin/sh

python3 Optimisation.py -i 499 -p 30 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'newbuild_noSSS' -b 'noBatteries' -m 0.6
python3 Optimisation.py -i 499 -p 10 -e 99 -n 'Mekong_Grid' -s 'no_nuclear' -f 'newbuild_noSSS' -b 'batteries' -m 0.6

