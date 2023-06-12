#!/bin/bash

RELDIR="dkpn_v${1}"
mkdir ${RELDIR}
cp *py ${RELDIR}/
cp -r dkpn ${RELDIR}/
cp example_bash.sh ${RELDIR}

zip -r ${RELDIR}.zip ${RELDIR}
mv ${RELDIR}.zip releases/
rm -r ${RELDIR}

echo "DONE: ${RELDIR}"
