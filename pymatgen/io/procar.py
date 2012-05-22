import os
import sys
import re
import time
import logging
import glob
from optparse import OptionParser
from pprint import pprint as pp

import pylab
from scipy.ndimage.filters import gaussian_filter1d

from pymatgen.io.vaspio import Poscar

log = logging.getLogger('procar')


class Kpoint(object):
    def __init__(self, kpoint_id, abc, weight, eigenvalues):
        self.kpoint_id = kpoint_id
        self.abc = abc
        self.weight = weight
        self.eigenvalues = eigenvalues

    @property
    def a(self):
        return self.abc[0]

    @property
    def b(self):
        return self.abc[1]

    @property
    def c(self):
        return self.abc[2]

    def __str__(self):
        abc = " ".join(["{i:.3f}".format(i=i) for i in self.abc])
        return "id : {i} abc: [{abc}] weight : {w:.4f} neigenvalues {n}".format(i=self.kpoint_id, abc=abc, w=self.weight, n=len(self.eigenvalues))

    def __repr__(self):
        return str(self)

class Eigenvalue(object):
    def __init__(self, band_id, energy, occu, sites):
        self.band_id = band_id
        self.energy = energy
        self.occu = occu
        self.sites = sites

    def __str__(self):
        return "band_id {i} energy {e:.4f} occu {o:.4f} nsites {n}".format(i=self.band_id, e=self.energy, o=self.occu, n=len(self.sites))

    def __repr__(self):
        return str(self)

class OrbitalSite(object):
    MOMENTS = 's py pz px dxy dyz dz2 dxz dx2 p d eg t2g total'.split()
    # dx2 is shorthand for dx2-y2
    def __init__(self, site_id, element, s, py, pz, px, dxy, dyz, dz2, dxz, dx2):
        self.site_id = site_id
        self.element = element
        self.s = s
        self.px = px
        self.py = py
        self.pz = pz
        self.dxy = dxy
        self.dyz = dyz
        self.dz2 = dz2
        self.dxz = dxz
        self.dx2 = dx2

    @property
    def p(self):
        return self.px + self.py + self.pz

    @property
    def d(self):
        return self.t2g + self.eg

    @property
    def t2g(self):
        return self.dxy + self.dxz + self.dyz

    @property
    def eg(self):
        return self.dx2 + self.dz2

    @property
    def total(self):
        moments = 's p d'.split()
        return sum([getattr(self, m) for m in moments])

class Dos(object):
    def __init__(self, up, down=None, efermi=None):
        """
        Args:
            up, down: an array of Kpoint instances

        if only up is given, then the system is assumed
        to be non spin-polarized.
        """
        self._up = up
        self._down = down
        self._efermi = efermi

    @property
    def up(self):
        return self._up

    @property
    def down(self):
        return self._down

    @property
    def efermi(self):
        """
        Determine Fermi Level from occupations
        """
        if self._efermi is None:
            return self._get_efermi()
        else:
            return self._efermi

    def summary(self):
        outs = []
        outs.append("Density of States")
        if self.down is not None:
            outs.append("Spin polarized*")
        outs.append("nsites   : {n}".format(n=self.nsites))
        outs.append("nkpoints : {n}".format(n=len(self.up)))
        outs.append("elements : {n}".format(n=self.elements))
        outs.append("Fermi    : {e:.4f}".format(e=self.efermi))
        #outs.append("nbands   : {n}".format(n=len(self.up[0]eigenvalues)))

        return "\n".join(outs)

    @property
    def nsites(self):
        """
        Get the number of sites
        """
        return len(self._up[0].eigenvalues[0].sites)

    @property
    def elements(self):
        return [s.element for s in self._up[0].eigenvalues[0].sites]

    def _get_efermi(self):
        """ Get Fermi Level from occupations"""

        def get_max_energy(kpoints):
            efermi = -1000000
            for kpoint in kpoints:
                for eigenvalue in kpoint.eigenvalues:
                    if eigenvalue.occu > 0.0:
                        if eigenvalue.energy > efermi:
                            efermi = eigenvalue.energy
            return efermi

        #Non-Spin polarized
        if self.down is None:
            return get_max_energy(self.up)
        else:
            efermis = [get_max_energy(kpt) for kpt in [self.up, self.down]]
            return max(efermis)

    def get_band_gap(self):
        """
        Explicitly compute this from the occupations
        """
        def get_cbm_energy(kpoints):
            cbm_energy = 1000000
            for kpoint in kpoints:
                for eigenvalue in kpoint.eigenvalues:
                    if eigenvalue.occu == 0.0:
                        if eigenvalue.energy < cbm_energy:
                            cbm_energy = eigenvalue.energy
            return cbm_energy

        #Non-Spin polarized
        if self.down is None:
            return get_cbm_energy(self.up)
        else:
            energies = [get_cbm_energy(kpt) for kpt in [self.up, self.down]]
            return min(energies)


    def compute_by_site_range(self, start_index, final_index, moment='total', spin='total'):
        raise NotImplemented()

    def compute_dos_by_elements(self, elements, spin='up', zero_to_efermi=True):
        """
        Method get sum by Element

        Args:

            elements: str, or array of str

        """

        if isinstance(elements, str):
            elements = [elements]

        # 'up' is the up or total for non-spinpolarized systems
        # 'down' is the down states
        # 'total' is the sum of the up and the down

        spin_map = {'up': [self.up], 'down': [self.down], 'total': [self.up, self.down]}

        spin_channels = spin_map[spin]
                
        # energy grid to compute the DOS on
        dos_grid = []
        de = 0.05
        
        # these should really be determined by the max and min eigenvalues
        initial_e = -30
        final_e = 30
        
        nsteps = int((final_e - initial_e) / de)
        energy_grid = (i * de + initial_e for i in xrange(nsteps))
        
        for energy in energy_grid:
            tmp = {}
            
            # Initilize dict
            for moment in OrbitalSite.MOMENTS:
                tmp[moment] = 0.0
            
            for spin_channel in spin_channels:
                for kpoint in spin_channel:
                    for eigenvalue in kpoint.eigenvalues:
                        
                        e_max = energy + de
                        
                        if (eigenvalue.energy > energy and eigenvalue.energy <= e_max):
                            for site in eigenvalue.sites:
                                if site.element in elements:
                                    for moment in OrbitalSite.MOMENTS:
                                        tmp[moment] = tmp[moment] +  getattr(site, moment) * (1 / de) * kpoint.weight
            
            # Create an array of e, s, px, py, pz, p, etc...
            if zero_to_efermi:
                dos = [energy - self.efermi]
            else:
                dos = [energy]

            for moment in OrbitalSite.MOMENTS:
                dos.append(tmp[moment])
            dos_grid.append(dos)
        return dos_grid

class ProcarParser(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.site_id_to_element = {}

    def parse_b_block(self, b_block):
        """docstring for parse_b_block"""
        lines = b_block.split("\n")

        #print "Band block"
        #pp(lines)
        #print (lines, )

        #rion = re.compile(r'\d+\.\d+\s+\d+\.\d+\.*')
        #rion = re.compile("\\d+\\.\\d+\\s+\\d+\\.\\d+\\.*")
        rion = re.compile(r'^\s+\d+\s+\d\.\d+')

        renergy = re.compile('energy')

        band = dict(sites=[])
        eigenval = Eigenvalue(0, 0.0, 0.0, [])
        #print len(lines)

        for i, line in enumerate(lines):
            #print (i, line)
            if renergy.search(line):
                d = line.split()
                band_id, energy, occu = int(d[0]), float(d[3]), float(d[6])
                #print ('Energy', band_id, energy, occu)
                #band['band_id'] = band_id
                #band['energy'] = energy
                #band['occu'] = occu
                eigenval = Eigenvalue(band_id, energy, occu, [])

            if rion.search(line):
                #print ("ION", d)
                d = line.split()
                # the site_id start at 1, not 0
                site_id = int(d[0]) - 1
                # total is the last one, exclude it
                data = map(float, d[1:-1])

                if site_id in self.site_id_to_element:
                    element = self.site_id_to_element[site_id]
                else:
                    #pp(site_id)
                    #print "Default to 'H'"
                    element = "H"

                site = OrbitalSite(site_id, element, *data)

                eigenval.sites.append(site)

        return eigenval

    def parse_k_block(self, k_block):
        band_blocks = k_block.split('band')
        header = band_blocks[0].split()

        #print ('header', header)

        k_id, weight = int(header[0]), float(header[-1])

        if len(header) == 8:
            # ' 158 :   -0.45833333 0.45833333 0.45833333     weight = 0.00231481\n\n
            abc = map(float, header[2:5])
        else:
            #FIXME BUGGG
            # The data sometimes doesn't have spaces!
            # k-point   53 :   -0.45833333-0.45833333 0.04166667     weight = 0.00462963
            #abc = header
            abc = [0, 0, 0]

        #print (k_id, abc, weight)
        #kpoint = dict(kpoint_id=k_id, abc=abc, weight=weight, bands=[])

        #kpoint = Kpoint(k_id, abc, weight, [])

        #print len(band_blocks)

        eigenvalues = []
        for band_block in band_blocks[1:]:
            eigenval = self.parse_b_block(band_block)
            eigenvalues.append(eigenval)

        kpoint = Kpoint(k_id, abc, weight, eigenvalues)
        return kpoint

    def parse(self):
        """
        Core parsing routine

        - Looks for POSCAR to get element symbols

        """

        dir_name = os.path.dirname(os.path.abspath(self.file_name))

        poscar_file_name = os.path.join(dir_name, 'POSCAR')

        if os.path.exists(poscar_file_name):
            #print "Found POSCAR"
            poscar = Poscar.from_file(poscar_file_name)
            self.site_id_to_element = {i:site.specie.symbol for i, site in enumerate(poscar.struct.sites)}
        else:
            self.site_id_to_element = {}

        with open(self.file_name, 'r') as f:
            text = f.read()

        rx = re.compile(' k-point ')
        kpoint_blocks = rx.split(text)
        # first entry has the header

        # '# of k-points:  159         # of bands:  21         # of ions:   2\n'
        header = kpoint_blocks[0].split()
        nkpoints, nbands, nsites = int(header[6]), int(header[10]), int(header[-1])

        #print (nkpoints, nbands, nsites)

        kpoints = []

        for kpoint_block in kpoint_blocks[1:]:
            kpoint = self.parse_k_block(kpoint_block)
            kpoints.append(kpoint)

        del text
        del kpoint_blocks
        # If the actual found kpoints == nkpionts * 2
        # Then the run was spin polarized

        actual_nkpoints = len(kpoints)
        n = actual_nkpoints / 2

        #print (nkpoints, actual_nkpoints)
        up = kpoints[0:n]
        down = kpoints[n:]
        dos = Dos(up, down)
        return dos

class DosPlotter(object):

    def __init__(self, dos):
        self.dos = dos

    def plot_total(self, file_name, title=None):
        # Plot element's by total DOS on each site
        # returns a Plot object?
        # grab the DosGrid object from the plot?
        elements = self.dos.elements

        plot_element_up = self.dos.compute_dos_by_elements(elements, spin='up')

        if not self.dos.down is None:
            plot_element_down = self.dos.compute_dos_by_elements(elements, spin='down')

        #pp(plot_element_total)

        #print len(plot_element_total[0])

        moment_map = {moment: i for i, moment in enumerate(OrbitalSite.MOMENTS)}

        plot_moments = 's p d total'.split()
        color_map = dict(zip(plot_moments, 'g r b c'.split()))

        for moment in plot_moments:
            i = moment_map[moment]
            x = [e[0] for e in plot_element_up]
            y = [e[i + 1] for e in plot_element_up]
            label = moment
            #pylab.plot(x, y, label=label, color=color_map[moment])

        sigma = 0.06

        #UP
        for moment in plot_moments:
            i = moment_map[moment]
            x = [e[0] for e in plot_element_up]
            y = [e[i + 1] for e in plot_element_up]
            smeared_y = gaussian_filter1d(y, sigma / 0.05)
            label = moment
            pylab.plot(x, smeared_y, label=label, color=color_map[moment])

        # DOWN
        if not self.dos.down is None:
            for moment in plot_moments:
                i = moment_map[moment]
                x = [e[0] for e in plot_element_down]
                y = [e[i + 1] * -1 for e in plot_element_down]
                smeared_y = gaussian_filter1d(y, sigma / 0.05)
                label = moment
                pylab.plot(x, smeared_y, label=None, color=color_map[moment])

        label_size = 20
        pylab.rc('text', usetex=True)

        if not title is None:
            pylab.title(r'$\mathrm{' + '\ task\_id\ '.join(title.split('_')) + '}$')

        pylab.ylabel(r'$\mathrm{DOS\ (States/eV/Spin)}$', fontsize=label_size)
        pylab.xlabel(r'$\mathrm{E-E_f\ (eV)}$', fontsize=label_size)
        pylab.xlim([-10, 10])
        pylab.axvline(0.0, color='k')

        pylab.legend(loc='upper right')
        print "Saving plot {f}".format(f=file_name)
        pylab.savefig(file_name)
        pylab.clf()

    def plot_by_elements(self, file_name, moments=None, title=None):
        """

        Args:
            file_name (str) of filename to write to
            moments (dict) of {element:moments} {"O":['p', 'total'], "Mn":['t2g', 'eg'] } 
            title (str) title of plot

        If moments is None, the total dos is plotted

        """
        # Plot element's by total DOS on each site
        # returns a Plot object?
        # grab the DosGrid object from the plot?
        elements = list(set(self.dos.elements))

        # A Map of moments to ints in the dos array
        moment_map = {moment: i for i, moment in enumerate(OrbitalSite.MOMENTS)}

        # Plot The total for each element

        plot_moments = ['total']

        #color_map = {'total': 'g'}
        # Colors can be specified as html colors
        # http://www.webstandards.org/learn/reference/charts/color_names/

        colors = 'blue red fuchsia green aqua purple lime maroon navy teal'.split()

        color_map = dict(zip(elements, colors))
        
        for element in elements:
        
            plot_element_up = self.dos.compute_dos_by_elements(element, spin='up')

            if not self.dos.down is None:
                plot_element_down = self.dos.compute_dos_by_elements(element, spin='down')

            sigma = 0.06

            #UP
            for moment in plot_moments:
                i = moment_map[moment]
                x = [e[0] for e in plot_element_up]
                y = [e[i + 1] for e in plot_element_up]
                smeared_y = gaussian_filter1d(y, sigma / 0.05)
                pylab.plot(x, smeared_y, label=element, color=color_map.get(element, 'blue'))

            # DOWN
            if not self.dos.down is None:
                for moment in plot_moments:
                    i = moment_map[moment]
                    x = [e[0] for e in plot_element_down]
                    y = [e[i + 1] * -1 for e in plot_element_down]
                    smeared_y = gaussian_filter1d(y, sigma / 0.05)
                    pylab.plot(x, smeared_y, label=None, color=color_map.get(element, 'blue'))

        # Pretty up plot
        label_size = 20
        pylab.rc('text', usetex=True)

        if not title is None:
            pylab.title(r'$\mathrm{' + '\ task\_id\ '.join(title.split('_')) + '}$')

        pylab.ylabel(r'$\mathrm{DOS\ (States/eV/Spin)}$', fontsize=label_size)
        pylab.xlabel(r'$\mathrm{E-E_f\ (eV)}$', fontsize=label_size)
        pylab.xlim([-10, 10])
        pylab.axvline(0.0, color='k')

        pylab.legend(loc='upper right')
        print "Saving plot {f}".format(f=file_name)
        pylab.savefig(file_name)
        pylab.clf()


def test_procar_parser():
    """Basic smoke test for ProcarParser. Still very slow"""
    file_name = ''
    parser = ProcarParser(file_name)
    dos = parser.parse()
    print dos


def run(procar_file_name, output_file=None, plot_element=False):
    """
    Core routine to compute the DOS
    """
    title = None
    t0 = time.time()
    parser = ProcarParser(procar_file_name)
    dos = parser.parse()
    run_time = time.time() - t0
    print "Parsing took {s} sec ({m} min)".format(s=int(run_time), m=int(run_time / 60.0))
    dos_plotter = DosPlotter(dos)

    if plot_element:
        dos_plotter.plot_by_elements(output_file)
    else:
        dos_plotter.plot_total(output_file)


def main():
    " Main Point of Entry"

    parser = OptionParser()

    parser.add_option('-e', '--element', dest='plot_element', action='store_true', help='Element Symbol (e.g., Fe)')
    #parser.add_option('-t', '--total', dest='total', help='Plot total')
    parser.add_option('-o', '--output', dest='output_file', default=None, help='Output file to write plot to.')

    opts, args = parser.parse_args()

    print (opts, args)

    if len(args) != 1:
        print "Provide a path to a PROCAR file"
        return -1
    else:
        procar_file_name = args[0]
        run(procar_file_name, opts.output_file, opts.plot_element)
        return 0

if __name__ == '__main__':
    sys.exit(main())

