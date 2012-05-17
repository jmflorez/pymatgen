#!/usr/bin/env python

import os
import sys
import re
import shutil
import tempfile
import subprocess

from pymatgen.command_line.utils import which

from pymatgen.io.vaspio import Poscar
#from pymatgen.core.structure import Structure
from pymatgen.core.structure_modifier import StructureEditor

BADER_EXE = 'bader'

#commandline utils
def call_bader(file_name):
    """
    1. Create a temp directory
    2. cd tmpdir
    2. call bader
    3. Parse ACF.dat file
    """
    if which(BADER_EXE) is None:
        print "Unable to find {e}".format(e=BADER_EXE)
        print "Please download the code from http://theory.cm.utexas.edu/bader/ and add the executable ({e}) to your PATH.".format(e=BADER_EXE)
        return None

    dir_name = os.path.dirname(os.path.abspath(file_name))
    temp_dir = tempfile.gettempdir()

    #print temp_dir

    os.chdir(temp_dir)

    cmd = [BADER_EXE, file_name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

    #ACF.dat
    #Bader charge and Volume
    # File has the format

    #       #         X           Y           Z        CHARGE     MIN DIST    ATOMIC VOL
    # --------------------------------------------------------------------------------
    #    1      2.5633      2.5633      2.5633      7.1291      1.3282     21.9284
    #    2      0.0000      0.0000      0.0000     11.8709      1.0966     11.7555
    # --------------------------------------------------------------------------------
    #VACUUM CHARGE:               0.0000
    #VACUUM VOLUME:               0.0000
    #NUMBER OF ELECTRONS:        19.0000

    acf_file_name = os.path.join(temp_dir, 'ACF.dat')

    data = []
    rx = re.compile(r'^\s*\d')
    if os.path.exists(acf_file_name):

        with open(acf_file_name) as f:
            lines = f.readlines()

        for line in lines:
            if rx.search(line):
                d = line.split()
                index = int(d[0])
                charge = float(d[4])
                volume = float(d[-1])
                bader = {'charge': charge, 'volume': volume}
                data.append(bader)

    # Delete tmp dir
    #shutil.rmtree(temp_dir)
    #change back to starting working directory
    os.chdir(dir_name)
    return data


class Bader(object):
    def __init__(self, charge, volume):
        self.charge = charge
        self.volume = volume

    @property
    def to_dict(self):
        return dict(charge=self.charge, volume=self.volume)

    @staticmethod
    def from_dict(d):
        return Bader(d['charge'], d['volume'])


class BaderAnalysis(object):
    """
    Bader analysis for VASP CHGCAR
    """
    def __init__(self, file_name):
        # CHGCAR file
        self.file_name = file_name
        self._structure = None

    @property
    def structure(self):
        if self._structure is None:
            return self._parse()
        else:
            return self._structure

    def _parse(self):
        """
        1. Call bader CHGCAR
        2. Parse output
        3. build Structure(site_properties={'bader': []})

        """
        # Returns an array of [{charge:float, volume:float}, {}]
        data = call_bader(self.file_name)

        #parse CHGCAR for Structure
        poscar = Poscar.from_file(self.file_name)

        structure = poscar.struct
        structure_editor = StructureEditor(structure)

        #FIXME Temporary hack to get Structure to support bader properties
        site_property_name = 'bader'
        site_property_name = 'charge'

        structure_editor.add_site_property(site_property_name, data)
        self._structure = structure_editor.modified_structure

        return self._structure

    def all_charges_by_element(self, element_name):
        return [site.charge.get('charge', 0.0) for site in self.structure.sites if site.specie.symbol == element_name]

    def average_charge_by_element(self, element_name):
        "Determine the average Bader"
        c = self.structure.composition.to_dict
        n = c[element_name]
        total_charge = self.total_charge_by_element(element_name)
        return total_charge / n

    def total_charge_by_element(self, element_name):
        return sum(self.all_charges_by_element(element_name))

    def plot_by_element(self, element_name, file_name=None):
        "Generate a Histogram of the Bader charges for element symbol"

        charges = self.all_charges_by_element(element_name)

        import pylab
        pylab.rc('text', usetex=True)        
        pylab.title(r'$' + self.structure.composition.reduced_formula + '$')
        n, bins, patches = pylab.hist(charges, 25, normed=True, histtype = 'bar', facecolor='blue', alpha=0.6)
        
        pylab.xlabel("{e} Bader Charge".format(e=element_name))
        pylab.grid(True)
        if file_name is None:
            pylab.show()
        else:
            pylab.savefig(file_name)

        pylab.clf()

