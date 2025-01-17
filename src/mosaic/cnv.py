import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from h5.constants import CHROM

from mosaic.assay import _Assay
from mosaic.constants import COLORS, GENE_NAME, NORMALIZED_READS, PLOIDY, READS
from mosaic.plotting import plt, require_seaborn, sns


class Cnv(_Assay):
    """
    Container for CNV data.

    Inherits most methods from :class:`mosaic.assay._Assay`.
    See that for the documentation on other methods and visualizations.

    .. rubric:: Algorithms
    .. autosummary::

       get_gene_names
       name_id_by_pos
       normalize_reads
       compute_ploidy

    .. rubric:: CNV specific visualizations
    .. autosummary::

       plot_ploidy
       amp_uniformity
       cell_uniformity

    .. rubric:: Extended methods
    .. autosummary::

       heatmap
    """

    assay_color = COLORS[3]

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: list
            To be passed to mosaic.assay object.

        kwargs: dict
            To be passed to mosaic.assay object.
        """
        super().__init__(*args, **kwargs)

    def name_id_by_pos(self):
        """
        Renames the ids using the chromosome and position
        """
        if ('CHROM' in self.col_attrs) & ('start_pos' in self.col_attrs):
            self.col_attrs['id'] = np.array([c + ':%10.0f' % p for c, p in zip(self.col_attrs['CHROM'], self.col_attrs['start_pos'])])

        working_amplicons = np.where(self.layers[READS].sum(axis=0) > 0)[0]
        self.select_columns(working_amplicons)

    def normalize_reads(self, method = 'mb', diploid_cells =None):
        """
        Normalize the read counts.

        This normalization is used to determine
        the ploidy of the cells.

        @HZ: per Matt, normalization needs to be copy number aware, and should only use diploid cells
        
        Parameters
        ----------
        method: 'mb' or 'hz'
            Use 'hz' to use copy number aware normalization method

        diploid cells: list-like
            To be used as diploid baseline for normalization

        """

        if method == 'hz':
            if not isinstance(diploid_cells, np.ndarray):
                raise ValueError('Please provide the barcodes of diploid cells')

            rc = self.get_attribute('read_counts', constraint='row')
            normal_rc = rc.loc[diploid_cells, :]

            rc /= normal_rc.mean(axis=0) + 1
            rc /= (np.array(rc.mean(axis=1))[:, np.newaxis]) + 1
            rc *= 2
            normal_counts = rc.to_numpy()

        elif method == 'mb':
            dat = self.layers['read_counts']

            # Find good barcodes: those with at least 1/10th the reads of the 10th highest barcode
            dat_sum = dat.sum(axis=1)
            s = np.sort(dat_sum)[::-1]
            good = dat_sum > s[10] / 10

            print(f'Using {good.shape[0]} / {dat.shape[0]} cells for normalization')

            normal_counts = deepcopy(dat)

            # normalize by per-cell mean amplicon read depth
            normal_counts = normal_counts / (normal_counts.mean(axis=1)[:, None] + 1) 
            # normalize by per-amplicon median (for good cells only)
            normal_counts = normal_counts / (np.median(normal_counts[good, :], axis=0)[None, :] + 0.05)

        normal_counts = normal_counts * 2  # For diploid

        self.add_layer(NORMALIZED_READS, normal_counts)

    def compute_ploidy(self, diploid_cells, normalized_counts=NORMALIZED_READS):
        """
        Computes the ploidy of the cells

        This method adds a layer called 'ploidy'

        Parameters
        ----------
        diploid_cells : list-like
            The barcodes which are known to be diploid.
            All barcodes are normalized to the median
            counts of these barcodes for each amplicon.

        normalized_counts : str / np.array, default: :attr:`constants.NORMALIZED_READS`
            Name of the layer which is to be used
            for ploidy calculations. A numpy array
            with the same shape as the assay can
            also be passed.

        Raises
        ------
        ValueError
            When the reads have not yet been normalized.
            :meth:`Cnv.normalize_reads` has to be executed
            before computing ploidy.
        """

        normalized_counts = self.get_attribute(normalized_counts, constraint='row+col')

        diploid_cells = normalized_counts.loc[diploid_cells, :]
        diploid_cells_median = diploid_cells.median() # per-amplicon median coverage for each diploid cell
        diploid_cells_median[diploid_cells_median == 0] = 1

        ploidy = 2 * normalized_counts / diploid_cells_median
        self.add_layer(PLOIDY, ploidy.values)

    def get_gene_names(self, amplicon_gene_map_file: str = None, gene_name_col: str = GENE_NAME):
        """
        Get the gene names for each amplicon. If a mapping file is provided it is prioritized; otherwise gene names are fetched from Ensembl. The data is stored in the "gene_name" column attribute.

        For the mapping file, it is required to have the amplicon numbers (e.g. `AMPL00001`) in the first column, as this will be read as the index; it is preferred to store the gene names in the "gene_name" column. Optionally, cytoband information could be provided in the "cytoband" column.

        Inputs:

        amplicon_gene_map_file: str
            Path to the amplicon gene map file. Should be in tab-delimited column with first column as amplicon ID's, providing mapping between amplicon ID's and gene names. Default: None.
        
        gene_name_col: str
            Name of the column in the amplicon gene map file. Default: :attr:`constants.GENE_NAME`

        """
        print('Fetching annotation for amplicons.')
        
        # @HZ 07/09/2022
        if amplicon_gene_map_file is not None:
            amplicon_gene_map_df = pd.read_csv(amplicon_gene_map_file, index_col = 0, engine='python')
            amplicon_list = self.ids()
            mapped_gene_list = amplicon_gene_map_df.loc[amplicon_list, gene_name_col].values
            self.add_col_attr(GENE_NAME, mapped_gene_list)

            print(f'[get_gene_names] Added gene names to the "{GENE_NAME}" column attribute.')

            if 'cytoband' in amplicon_gene_map_df.columns:
                mapped_cytoband_list =  amplicon_gene_map_df.loc[amplicon_list, 'cytoband'].values
                self.add_col_attr('cytoband', mapped_cytoband_list)
                print(f'[get_gene_names] Added cytoband information to the "cytoband" column attribute.')

        else:
            if 'genome_version' in self.metadata:
                if isinstance(self.metadata['genome_version'], str):
                    if self.metadata['genome_version'] != 'hg19':
                        raise NotImplementedError('Annotation available only for the hg19 genome')
                else:
                    if not (self.metadata['genome_version'] == 'hg19').all():
                        raise NotImplementedError('Annotation available only for the hg19 genome')

            chrom = self.col_attrs[CHROM]
            start = self.col_attrs['start_pos']
            end = self.col_attrs['end_pos']
            regions = pd.DataFrame([chrom, start, end], index=['chrom', 'start', 'end']).T

            # Sort by position
            order = self._get_amplicon_order()

            # Group regions for fewer API calls
            group = regions.loc[order, :].reset_index(drop=True)
            i, j = 0, 1
            while j < group.shape[0]:
                r1 = group.loc[i, :]
                r2 = group.loc[j, :]
                if r2['chrom'] == r1['chrom'] and (r2['end'] - r1['start']) < 5 * (10 ** 6):  # Ensembl has a 5Mb limit
                    group.loc[i, 'end'] = r2['end']
                    group = group.drop(j).reset_index(drop=True)
                else:
                    i += 1
                    j += 1

            regs = np.array([g[0] + ':' + str(g[1]) + '-' + str(g[2]) for g in group.values])

            # Call the Ensembl API
            with ThreadPoolExecutor(max_workers=10) as pool:
                resps = list(pool.map(self._get_ensembl_gene, regs))

            data = []
            for resp in resps:
                for r in resp:
                    data.append([r['external_name'], r['seq_region_name'], r['start'], r['end']])
            data = pd.DataFrame(data, columns=['gene', 'chrom', 'start', 'end'])
            data = data.drop_duplicates()

            # Assign gene names to each region
            genes = []
            for c, s, e in zip(chrom, start, end):
                filt = (data['chrom'] == c) & (data['start'] < e) & (s < data['end'])
                g = data.loc[filt, 'gene']
                if len(g) == 0:
                    genes.append('Unknown')
                else:
                    genes.append('/'.join(g.values))

            self.add_col_attr(GENE_NAME, np.array(genes))
            print(f'Added gene names to the "{GENE_NAME}" column attribute')

    def _get_amplicon_order(self):
        """
        Amplicons ordered by position

        Returns
        -------
        np.array
            index of the ids to get the
            sorted amplicon list
        """
        ch = self.col_attrs[CHROM].copy()
        ch[ch == 'X'] = 23
        ch[ch == 'Y'] = 24
        ch = ch.astype(int)
        sp = self.col_attrs['start_pos'].copy()
        sp = sp.astype(int)
        order = (ch * 10 ** 15 + sp).argsort()

        return order

    def _get_ensembl_gene(self, region):
        """
        Get the gene name based on a region

        Parameters
        ----------
        region : str
            The region should be formated as
            {chrom}:{start}-{end} and "chrom"
            must not contain chr i.e. '11'
            instead of 'chr11' is expected.
        """

        server = 'https://grch37.rest.ensembl.org'
        ext = f"/overlap/region/human/{region}?feature=gene"

        while True:
            resp = requests.get(server + ext, headers={"Content-Type": "application/json"})
            if not resp.ok:
                time.sleep(0.2)
            else:
                break

        return resp.json()

    # --------- Plotting

    def _get_ids_from_features(self, features):
        """
        Identifies the ids for the given features

        Parameters
        ----------
        features : str / list-like
            Besides the accepted values, 'features' can
            also be one of {'genes', 'positions'} or a list
            of chromosomes e.g. ['1', '7', 'X'] or a list
            of gene names eg. ['BRAF', 'TP53']

        Returns
        -------
        np.ndarray
            A list of ids
        """

        if features is None:
            return None, None

        error_msg = '"features" could not be identified. '

        if GENE_NAME not in self.col_attrs:
            raise ValueError('Amplicon annotation not found! Please run "sample.cnv.get_gene_names()" to add gene names to col_attrs.')
            # self.get_gene_names()

        if features == 'genes' or all([i in self.col_attrs[GENE_NAME] for i in features]):
            genes = self.col_attrs[GENE_NAME].copy()
            # order = genes.argsort()
            # @HZ 07/13/2022 -- might be better to sort by genomic coordinates
            order = self._get_amplicon_order() 
            ids = self.ids()[order]
            genes = self.col_attrs[GENE_NAME][order]
            if len(features) > 1: # focus on particular genes
                mask = np.isin(genes, features)
                genes = genes[mask]
                ids = ids[mask]
            else:
                mask = np.array([True] * len(genes))
            chroms = 'chr' + self.col_attrs[CHROM][order]
            genes = np.char.add(genes, np.full_like(genes, '<br>'))
            genes = np.char.add(genes, chroms[mask].astype(str))
            if 'cytoband' in self.col_attrs:
                cytobands = self.col_attrs['cytoband'][order]
                genes += ' ' + cytobands[mask]
                
            return ids, genes

        elif features == 'positions':
            order = self._get_amplicon_order()
            ids = self.ids()[order]

            chroms = 'chr' + self.col_attrs[CHROM][order]
            # if 'cytoband' in self.col_attrs:
            #     cytobands = self.col_attrs['cytoband'][order]
            #     chroms += cytobands

            # if GENE_NAME not in self.col_attrs:
            #     print("[_get_ids_from_features] gene_name not found in amplicon column attribute. Using chromosome number.")

            #     return ids, chroms
            # else:
            #     print('[_get_ids_from_features] gene_name found in amplicon column attribute.' )
            #     gene_names = self.col_attrs[GENE_NAME][order]
            #     gene_names += '<br>' + chroms
            #     return ids, gene_names
            return ids, chroms

        elif isinstance(features, str):
            explaination = f"'features' must be list-like or one of {'genes', 'positions'}"
            raise ValueError(error_msg + explaination)

        else:
            IDS = 'ids'
            POS = 'positions'
            GEN = 'genes'

            overlap = {IDS: 0, POS: 0, GEN: 0}
            chroms = self.col_attrs[CHROM].copy()

            overlap[IDS] = np.isin(features, self.ids()).sum() / len(features)
            overlap[POS] = np.isin(features, chroms).sum() / len(features)

            if GENE_NAME not in self.col_attrs:
                overlap[GEN] = 0
            else:
                genes = self.col_attrs[GENE_NAME].copy()
                overlap[GEN] = np.isin(features, genes).sum() / len(features)

            identified_kind = None
            for kind, ovlap in overlap.items():
                if ovlap > 0:
                    if identified_kind is None:
                        identified_kind = kind
                    else:
                        explaination = f'Values overlap with {kind} and {identified_kind} both'
                        raise ValueError(error_msg + explaination)

            def __get_list(kind, comp_list):
                nonlocal features
                ids, feats = [], []
                for f in features:
                    ind = np.where(np.isin(comp_list, f))[0]
                    if len(ind) == 0:
                        explaination = f'It is most likely to be a list of {kind} but "{f}" was not found in that list. '
                        explaination += f'Expected one of {set(comp_list)}'
                        raise ValueError(error_msg + explaination)
                    ids.extend(self.ids()[ind])
                    feats.extend([f] * len(ind))

                ids = np.array(ids, dtype=object)
                feats = np.array(feats, dtype=object)
                return ids, feats

            if identified_kind == IDS:
                return features, features
            elif identified_kind == POS:
                ids, feats = __get_list(POS, chroms)
                return ids, 'chr' + feats
            elif identified_kind == GEN:
                return __get_list(GEN, genes)
            else:
                explaination = f"If it is a list of genes, run 'sample.cnv.get_gene_names()' and make sure the '{GENE_NAME}' column attribute exists"
                explaination += "If it is a list of chromosomes, do not include 'chr' in the name i.e. pass [1, 2] instead of [chr1, chr7]"
                raise ValueError(error_msg + explaination)

    def heatmap(self, attribute, splitby='label', features=None, bars_order=None, convolve=0, title=''):
        """
        Extends :meth:`_Assay.heatmap`

        Besides the accepted values, 'features' can
        also be one of {'genes', 'positions'} or a list
        of chromosomes e.g. ['1', '7', 'X'] or a list
        of gene names eg. ['BRAF', 'TP53']
        """

        # renamed `features` to `plot_feature_groups` to be more informative
        ids, plot_feature_groups = self._get_ids_from_features(features)

        fig = super().heatmap(attribute, splitby=splitby, features=ids, bars_order=bars_order, convolve=convolve, title=title)

        if plot_feature_groups is not None and set(plot_feature_groups) != set(fig.layout.xaxis2.ticktext):
            un, ind, cnts = np.unique(plot_feature_groups, return_index=True, return_counts=True)
            ticks = (ind + cnts / 2).astype(int)
            fig.layout.xaxis2.ticktext = plot_feature_groups[ticks]
            fig.layout.xaxis2.tickvals = ticks

            # @HZ 07/13/2022: when plotting chromosomal positions, we want to add the gene names to the hover labels
            if features == 'positions' and GENE_NAME in self.col_attrs.keys():
                order = self._get_amplicon_order()
                plot_feature_groups += '<br>' + self.col_attrs[GENE_NAME][order]
            
            fig.data[1].x = ids + '<br>' + plot_feature_groups

            for i in ind:
                fig.add_vline(i - 0.5, line_color='lightcyan', line_width=1, row=1, col=2)

        return fig

    # @HZ 12/13/2021: added in amplicon mapping option
    def plot_ploidy(self, cluster, features=None, amplicon_map=None):
        """
        Plots the ploidy of the cluster

        Parameters
        ----------
        cluster : str
            Barcodes belonging to this label
            will be fetched for the plot.

        features : list-like / {'genes'} default: None
            The features which are to be plotted.
            If `None`, then all the amplicons are
            plotted. This attribute can be used
            to change the order of the amplicons.

            If 'genes', then the amplicons are collapsed
            to their gene level.

        amplicon_map: a dictionary like {amplicon number: gene name}
            Required when features are a list of amplicons

        Raises
        ------
        ValueError:
            When the cluster is not found in the labels

        RuntimeError:
            If `ploidy` is not found in the layers.
            :meth:`Cnv.compute_ploidy` has to be executed
            before running this method.
        """

        if PLOIDY not in self.layers:
            raise RuntimeError('Run `compute_ploidy` first')

        bars = self.barcodes(cluster)
        if len(bars) == 0:
            raise ValueError(f'No barcodes corresponding the {cluster} label were found')

        if features == 'genes':
            if GENE_NAME not in self.col_attrs:
                print('Annotation not found.')
                self.get_gene_names()

            ploidy = self.get_attribute(PLOIDY, constraint='row+col')

            data = ploidy.loc[bars, :].median()
            data = pd.DataFrame(data, columns=['Median ploidy'])
            data.loc[:, 'gene'] = self.col_attrs[GENE_NAME]
            data = data.groupby('gene').median().reset_index()
            x, y = 'gene', 'Median ploidy'
        elif isinstance(features, str):
            raise ValueError("'features' must be a list of ids or 'genes'")
        else:
            if features != None and amplicon_map == None:
                raise ValueError('amplicon_map needs to be provided')

            if features != None and any(np.isin(features,list(amplicon_map.keys()))) == False:
                raise ValueError("not all input amplicons are in the amplicon map's keys")

            ploidy = self.get_attribute(PLOIDY, constraint='row+col', features=features)

            data = ploidy.loc[bars, :].median()
            data = pd.DataFrame(data, columns=['Ploidy']).reset_index()
            # if GENE_NAME in self.col_attrs:
            #     data.loc[:, 'gene'] = self.col_attrs[GENE_NAME]
            # else:
            #     print('Annotation not found. Run "sample.cnv.get_gene_names()" to add gene names.')
            if features != None:
                data.loc[:, 'gene'] = [amplicon_map[i] for i in features]
            else:
                data.loc[:, 'gene'] = self.col_attrs[GENE_NAME]
            #data = data.iloc[self._get_amplicon_order(), :]
            x, y = 'index', 'Ploidy'

        fig = px.scatter(data, x=x, y=y, template='gridon')

        fig.update_layout(width=1000,
                          height=600,
                          xaxis_title='',
                          xaxis_automargin=True,
                          title=cluster)

        fig.data[0].hovertemplate = '<b>%{y:.2f}</b><extra>%{x}</extra>'

        if features != 'genes' and 'gene' in data:
            fig.data[0].hovertemplate = '<b>%{y:.2f}</b><extra>%{x}<br>%{customdata}</extra>'
            fig.data[0].customdata = data['gene']

        if max(data[y]) < 4:
            fig.layout.yaxis.range = (0, 4)

        return fig

    @require_seaborn
    def amp_uniformity(self, title='', **kwargs):
        """
        Plot amplicon uniformity.

        This is a metric to determine the
        uniformity of the reads across amplicons.

        Parameters
        ----------
        title: str
            Appended to the sample name in the title.
        kwargs: dict
            Passed to the seaborn barplot.

        Returns
        -------
        ax : matplotlib.pyplot.axis
        """
        amp_reads = self.layers[READS]
        amp_names = self.ids()

        data = pd.DataFrame(amp_reads, columns=amp_names)
        data.loc['mean', :] = data.mean(axis=0)
        data = data.sort_values(axis=1, by=['mean'], ascending=False)
        data = data.drop(index=['mean'])

        threshold = 0.2 * data.mean().mean()
        num_cells = data.shape[0]
        num_amps = data.shape[1]
        data = data > threshold
        data = 100 * data.sum(axis=0) / num_cells

        sns.set(style='whitegrid', font_scale=1.5)
        plt.figure(figsize=(20, 10))

        ax = sns.barplot(data.index, data.values, linewidth=0, color=self.assay_color, **kwargs)

        # Loop over the bars, and adjust the width (and position, to keep the bar centered)
        # This is needed to avoid unequal spacing artifacts
        widthbars = np.array([1] * num_cells)
        for bar, newwidth in zip(ax.patches, widthbars):
            x = bar.get_x()
            width = bar.get_width()
            centre = x + width / 2.

            bar.set_x(centre - newwidth / 2.)
            bar.set_width(newwidth)

        if title == '':
            title = self.title + ' - '
        title += 'DNA amplicon uniformity per cell'

        ax.set_title(title, fontsize=19)
        ax.set_ylabel(f'Percentage cells above threshold\n{num_cells} cells', fontsize=19)
        ax.set_xlabel(f'{num_amps} amplicons sorted by mean reads', fontsize=19)
        ax.set_xticklabels('')

        description = f'Threshold is calculated as 0.2 * mean of reads per cell per amplicon.\n Here the threshold is {threshold:0.1f}.'
        ax.text(0.5, -0.12, description, transform=ax.transAxes, fontsize=12, ha='center')

        return ax

    @require_seaborn
    def cell_uniformity(self, title='', **kwargs):
        """
        Plot cell uniformity.

        A metric to determine the completeness of the data
        across the cells.

        Parameters
        ----------
        title : str
            Appended to the sample name in the title.
        kwargs : dict
            Passed to the seaborn barplot.

        Returns
        -------
        ax : matplotlib.pyplot.axis
        """

        amp_reads = self.layers[READS]
        amp_names = self.ids()

        data = pd.DataFrame(amp_reads, columns=amp_names)

        threshold = 0.2 * data.mean().mean()
        num_amps = data.shape[1]
        num_cells = data.shape[0]
        data = data > threshold
        data = 100 * data.sum(axis=1) / num_amps
        data = data.sort_values(ascending=False).reset_index(drop=True)
        cells_passing = data[data < 80].index[0]

        sns.set(style='whitegrid', font_scale=1.5)
        plt.figure(figsize=(20, 10))

        col = kwargs['color'] if 'color' in kwargs else self.assay_color
        col_desat = sns.desaturate(col, 0.3)
        colors = [col] * (cells_passing) + [col_desat] * (num_cells - cells_passing)
        ax = sns.barplot(data.index, data.values, palette=colors, linewidth=0, **kwargs)

        # Loop over the bars, and adjust the width (and position, to keep the bar centered)
        # This is needed to avoid unequal spacing artifacts
        widthbars = np.array([1] * num_cells)
        for bar, newwidth in zip(ax.patches, widthbars):
            x = bar.get_x()
            width = bar.get_width()
            centre = x + width / 2.

            bar.set_x(centre - newwidth / 2.)
            bar.set_width(newwidth)

        if title == '':
            title = self.title + ' - '
        title += 'Cell uniformity per amplicon'

        ax.set_title(title, fontsize=19)
        ax.set_ylabel(f'Percentage amplicons above threshold\n{num_amps} amplicons', fontsize=19)
        ax.set_xlabel(f'{num_cells} total cells of which {cells_passing} have more than 80% amplicons above threshold', fontsize=19)
        ax.set_xticklabels('')
        ax.axhline(80, color='black')
        ax.axvline(cells_passing, color='black')

        description = f'Threshold is calculated as 0.2 * mean of reads per cell per amplicon.\n Here the threshold is {threshold:0.1f}.'
        ax.text(0.5, -0.12, description, transform=ax.transAxes, fontsize=12, ha='center')

        return ax

    # @HZ 07/25/2022
    def update_coloraxis(self, fig):
        """
        Sets the colorscale for CNV when plotting NB_EM-calculated ploidy

        Parameters
        ----------
        fig : plotly.Figure
            The figure whose layout has to
            be updated.
        layer : str
            The layer according to which the
            coloraxis has to be updated.
        """
        pass

