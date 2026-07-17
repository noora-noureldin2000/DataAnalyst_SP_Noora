import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# ggsci-inspired journal color palettes
Lancet = ['#00468B', '#ED0000', '#42B540', '#0099B4', '#925E9F', '#FDAF91', '#AD002A', '#ADB6B6']
NEJM = ['#BC3C29', '#0072B5', '#E18727', '#20854E', '#7876B1', '#6F99AD', '#FFDC91', '#EE4C97']
JAMA = ['#374E55', '#DF8F44', '#00A1D5', '#B24745', '#79AF97', '#6A6599', '#80796B', '#F2B1C0']
Nature = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
ColorblindSafe = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']

PALETTES = {
    'lancet': Lancet,
    'nejm': NEJM,
    'jama': JAMA,
    'nature': Nature,
    'colorblind': ColorblindSafe,
}

def set_publication_style(palette='lancet', dpi=800, font_size=11, family='Arial'):
    rcParams['font.family'] = family
    rcParams['font.size'] = font_size
    rcParams['axes.titlesize'] = font_size + 2
    rcParams['axes.labelsize'] = font_size
    rcParams['xtick.labelsize'] = font_size - 2
    rcParams['ytick.labelsize'] = font_size - 2
    rcParams['legend.fontsize'] = font_size - 2
    # Apply dpi param to both display and export so the argument is fully respected
    rcParams['figure.dpi'] = min(dpi, 150)   # cap display DPI at 150 for screen usability
    rcParams['savefig.dpi'] = dpi
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.facecolor'] = 'white'
    rcParams['axes.facecolor'] = 'white'
    rcParams['axes.edgecolor'] = 'black'
    rcParams['axes.linewidth'] = 0.8
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linestyle'] = ':'
    rcParams['grid.color'] = '#CCCCCC'
    rcParams['legend.frameon'] = True
    rcParams['legend.edgecolor'] = 'lightgray'
    rcParams['legend.fancybox'] = False
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.size'] = 4
    rcParams['ytick.major.size'] = 4
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    mpl.rcParams['hatch.linewidth'] = 0.5
    palette_colors = PALETTES.get(palette.lower(), Lancet)
    rcParams['axes.prop_cycle'] = mpl.cycler(color=palette_colors)
    return palette_colors

def get_palette(name='lancet'):
    return PALETTES.get(name.lower(), Lancet)

def apply_figure_style(fig, title=None, xlabel=None, ylabel=None):
    for ax in fig.axes:
        if title and ax == fig.axes[0]:
            ax.set_title(title, fontweight='bold', pad=12)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

FIGURE_REVIEW_CHECKLIST = """
Figure Review Checklist
=======================
Before finalizing each figure, verify:

Typography:
  [ ] All text is readable (min 8pt for annotations, 10pt for labels)
  [ ] Font sizes follow hierarchy: title > axis labels > tick labels > annotations
  [ ] Use natural language labels (not variable names)
  [ ] Units are included where applicable (e.g., "Age (years)")

Colors:
  [ ] Colorblind-safe palette used (not default matplotlib)
  [ ] Muted, accessible colors — avoid neon/bright
  [ ] Alpha/transparency used for overlapping data points
  [ ] Sufficient contrast between categories

Layout:
  [ ] No overlapping text or data points
  [ ] Legend positioned optimally (top or bottom-right preferred)
  [ ] Margins are adequate — no clipping
  [ ] Axis ranges are appropriate (not misleading)

Theme:
  [ ] NOT using default matplotlib gray theme
  [ ] White/transparent background
  [ ] Grid lines are subtle (dashed, low alpha) or absent
  [ ] Top and right spines removed

Data:
  [ ] Point sizes ≥ 2.5 for scatter plots
  [ ] Line widths ≥ 0.8 for trend lines
  [ ] Error bars or confidence bands included where applicable
  [ ] Sample size (n) annotated where relevant
  [ ] Effect sizes or test statistics annotated where applicable

Export:
  [ ] Resolution ≥ 800 DPI for print, ≥ 300 DPI for review
  [ ] Format: PNG (review), TIFF LZW (journals), PDF/SVG (vectors)
  [ ] File named descriptively
"""
