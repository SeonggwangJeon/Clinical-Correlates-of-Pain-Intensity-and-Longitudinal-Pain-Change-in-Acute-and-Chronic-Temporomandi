"""Generate Supplementary Figures S2, S4, S5, S6 and interactive 3D HTML files.

Follows '20261119 Co-author figure instructions.docx' using the locked dataset
'data/20261119_TMD_MRI_89patients_CLEAN_deidentified.xlsx'.
"""

import io
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
import networkx as nx
import plotly.graph_objects as go
from PIL import Image, ImageChops

# ==============================================================================
# 1. Paths & Common Style (174 mm width, 600 dpi, Navy/Rose)
# ==============================================================================
DATA_PATH = 'data/20261119_TMD_MRI_89patients_CLEAN_deidentified.xlsx'
OUT_DIR = 'output/coauthor_figures'
os.makedirs(OUT_DIR, exist_ok=True)

INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e6e5e0'
NAVY = '#1F3A93'   # Acute
ROSE = '#D1495B'   # Chronic
CHARCOAL = '#333333'
LIGHT_GREY = '#BDBDBD'

MM = 1 / 25.4
W2 = 174 * MM  # full page width (6.85 inches)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.edgecolor': INK2,
    'axes.labelcolor': INK,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': GRID,
    'grid.linewidth': 0.5,
    'legend.frameon': False,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

LABELS = {
    'I_VAS': 'Initial VAS',
    'F_VAS': 'Final VAS',
    'I_CMI': 'Initial CMI',
    'F_CMI': 'Final CMI',
    'I_CMO': 'Initial CMO (mouth opening)',
    'I_TMJpain': 'TMJ pain',
    'I_SleepProblems': 'Sleep problems',
    'Onset_to_InitialVisit': 'Symptom duration (months)',
    'Treatment_Duration': 'Treatment duration (months)',
    'Age': 'Age (years)',
}

SHORT_LABELS = {
    'I_VAS': 'Initial VAS',
    'F_VAS': 'Final VAS',
    'I_CMI': 'Initial CMI',
    'F_CMI': 'Final CMI',
    'I_CMO': 'Initial CMO',
    'Onset_to_InitialVisit': 'Symptom duration',
    'Treatment_Duration': 'Treatment duration',
    'Age': 'Age',
}


def save_all(fig, stem):
    for ext in ['tiff', 'png', 'pdf']:
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tiff' else {}
        fig.savefig(os.path.join(OUT_DIR, f'{stem}.{ext}'), dpi=600, bbox_inches='tight', **kw)
    plt.close(fig)


def add_panel(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left', color=INK)


def trim_whitespace(im, pad=30):
    """Auto-crop excess white margins around Plotly 3D exports."""
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    diff = ImageChops.difference(im.convert('RGB'), bg.convert('RGB'))
    bbox = diff.getbbox()
    if not bbox:
        return im
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(im.width, bbox[2] + pad)
    y1 = min(im.height, bbox[3] + pad)
    return im.crop((x0, y0, x1, y1))


# ==============================================================================
# 2. Load Locked Dataset
# ==============================================================================
df = pd.read_excel(DATA_PATH, sheet_name='Data')
assert len(df) == 89 and df.isna().sum().sum() == 0


# ==============================================================================
# 3. Supplementary Figure S2: Group-specific coefficients
# ==============================================================================
s2_preds = ['I_CMI', 'I_CMO', 'I_TMJpain', 'I_SleepProblems']
fig, axes = plt.subplots(2, 2, figsize=(W2, W2 * 0.56))
fig.subplots_adjust(wspace=0.38, hspace=0.58)

for ax, pred, let in zip(axes.flat, s2_preds, 'abcd'):
    for grp_code, col, y_pos in [(0, NAVY, 1), (1, ROSE, 0)]:
        sub = df[df['Chronicity'] == grp_code]
        m = smf.ols(f'I_VAS ~ {pred} + Age + C(Sex)', data=sub).fit()
        b = m.params[pred]
        lo, hi = m.conf_int().loc[pred]
        ax.plot([lo, hi], [y_pos, y_pos], color=col, lw=1.6, solid_capstyle='round', zorder=2)
        ax.scatter(b, y_pos, s=34, color=col, edgecolor='white', lw=0.8, zorder=3)
        txt = f'{b:.2f} [{lo:.2f}, {hi:.2f}]'.replace('-', '−')
        ax.text(hi, y_pos + 0.24, txt, ha='right', va='bottom', fontsize=6.6, color=INK2)

    ax.axvline(0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks([0, 1], ['Chronic', 'Acute'], fontweight='bold', color=INK, fontsize=7.5)
    ax.set_ylim(-0.55, 1.65)
    ax.set_xlabel(f'β for {LABELS[pred]} (95% CI)', fontsize=7.5, fontweight='bold', color=INK)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    add_panel(ax, let, x=-0.18, y=1.04)

fig.legend(
    handles=[
        Line2D([], [], color=NAVY, lw=1.6, marker='o', ms=5, mec='white', label='Acute TMD (n = 51)'),
        Line2D([], [], color=ROSE, lw=1.6, marker='o', ms=5, mec='white', label='Chronic TMD (n = 38)'),
    ],
    loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05),
)
save_all(fig, 'Supplementary_Figure_S2_interaction_coefficients')


# ==============================================================================
# 4. Supplementary Figure S4: Correlation Matrices (Fixed annotations & layout)
# ==============================================================================
corr_vars = ['I_VAS', 'F_VAS', 'I_CMI', 'F_CMI', 'I_CMO', 'Onset_to_InitialVisit', 'Treatment_Duration', 'Age']
short_names = [SHORT_LABELS[c] for c in corr_vars]

fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.46), sharey=True)
fig.subplots_adjust(wspace=0.14, right=0.88)

im_ref = None
for idx, (ax, (grp_code, grp_name, let)) in enumerate(zip(axes, [(0, 'Acute TMD (n = 51)', 'a'), (1, 'Chronic TMD (n = 38)', 'b')])):
    mat = df[df['Chronicity'] == grp_code][corr_vars].corr().values
    im_ref = ax.imshow(mat, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')

    # Draw all 64 cell annotations explicitly (avoids seaborn/matplotlib 3.11 bug)
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            val = mat[i, j]
            txt_col = 'white' if abs(val) > 0.65 else INK
            ax.text(j, i, f'{val:.2f}'.replace('-', '−'), ha='center', va='center', fontsize=6.0, color=txt_col)

    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6.8, fontweight='bold', color=INK)
    ax.set_yticks(range(len(short_names)))
    if idx == 0:
        ax.set_yticklabels(short_names, fontsize=6.8, fontweight='bold', color=INK)
    else:
        ax.tick_params(axis='y', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(grp_name, loc='left', fontweight='bold', fontsize=8.5, color=INK, pad=8)
    add_panel(ax, let, x=-0.12 if idx == 1 else -0.32, y=1.04)

cbar_ax = fig.add_axes([0.90, 0.22, 0.018, 0.62])
cb = fig.colorbar(im_ref, cax=cbar_ax)
cb.set_label('Pearson r', fontsize=7.5, fontweight='bold', color=INK)
cb.ax.tick_params(labelsize=7, colors=INK2)
cb.outline.set_visible(False)

save_all(fig, 'Supplementary_Figure_S4_correlation_matrices')


# ==============================================================================
# 5. Supplementary Figure S5: 3D Correlation Networks (Zoomed & Cropped)
# ==============================================================================
def build_3d_network(target, html_name):
    sub = df[corr_vars]
    G = nx.Graph()
    for col in corr_vars:
        if col != target:
            G.add_edge(target, col, weight=sub[target].corr(sub[col]))
    others = [c for c in corr_vars if c != target]
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            G.add_edge(others[i], others[j], weight=sub[others[i]].corr(sub[others[j]]))

    pos = nx.spring_layout(G, dim=3, seed=42)

    edge_traces = []
    for u, v, d in G.edges(data=True):
        r = d['weight']
        edge_traces.append(go.Scatter3d(
            x=[pos[u][0], pos[v][0]],
            y=[pos[u][1], pos[v][1]],
            z=[pos[u][2], pos[v][2]],
            mode='lines',
            line=dict(
                color='#888888' if r >= 0 else '#555555',
                width=max(1.2, abs(r) * 8),
                dash='solid' if r >= 0 else 'dash',
            ),
            hoverinfo='text',
            text=f'{SHORT_LABELS[u]} – {SHORT_LABELS[v]}: r = {r:.3f}',
            showlegend=False,
        ))

    node_trace = go.Scatter3d(
        x=[pos[n][0] for n in G.nodes],
        y=[pos[n][1] for n in G.nodes],
        z=[pos[n][2] for n in G.nodes],
        mode='markers+text',
        text=[SHORT_LABELS[n] for n in G.nodes],
        textposition='top center',
        textfont=dict(size=22, color=INK, family='Arial'),
        marker=dict(
            size=[35 if n == target else 15 for n in G.nodes],
            color=[CHARCOAL if n == target else LIGHT_GREY for n in G.nodes],
            line=dict(width=1.5, color='white'),
        ),
        hoverinfo='text',
        showlegend=False,
    )

    fig3d = go.Figure(data=edge_traces + [node_trace])
    fig3d.update_layout(
        font=dict(color=INK, size=16, family='Arial'),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.05, y=1.15, z=0.65)),
        ),
    )
    fig3d.write_html(os.path.join(OUT_DIR, html_name))
    return fig3d


fig_s5a = build_3d_network('I_VAS', '3D_Net_IVAS.html')
fig_s5b = build_3d_network('F_VAS', '3D_Net_FVAS.html')

img_a = trim_whitespace(Image.open(io.BytesIO(fig_s5a.to_image(format='png', width=1400, height=1100, scale=3))), pad=40)
img_b = trim_whitespace(Image.open(io.BytesIO(fig_s5b.to_image(format='png', width=1400, height=1100, scale=3))), pad=40)
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.44))
fig.subplots_adjust(wspace=0.06)
for ax, im, let in zip(axes, [img_a, img_b], ['a', 'b']):
    ax.imshow(im)
    ax.axis('off')
    add_panel(ax, let, x=0.0, y=1.02)
fig.legend(
    handles=[
        Line2D([], [], marker='o', ls='', color=CHARCOAL, ms=7, label='Highlighted pain variable'),
        Line2D([], [], marker='o', ls='', color=LIGHT_GREY, ms=5.5, label='Other clinical variables'),
        Line2D([], [], color='#888888', lw=1.6, ls='-', label='Positive correlation (r > 0)'),
        Line2D([], [], color='#555555', lw=1.6, ls='--', label='Negative correlation (r < 0)'),
    ],
    loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=6.8,
)
save_all(fig, 'Supplementary_Figure_S5_3D_networks')


# ==============================================================================
# 6. Supplementary Figure S6: 3D Distributions (Zoomed & Cropped)
# ==============================================================================
def build_3d_space(target, html_name):
    fig3d = go.Figure()
    colors = {0: NAVY, 1: ROSE}
    surf_rgba = {0: 'rgba(31, 58, 147, 0.15)', 1: 'rgba(209, 73, 91, 0.15)'}
    names = {0: 'Acute', 1: 'Chronic'}

    for code in [0, 1]:
        sub = df[df['Chronicity'] == code]
        x = sub['Onset_to_InitialVisit'].values
        y = sub['Treatment_Duration'].values
        z = sub[target].values
        pts = np.vstack([x, y, z])

        fig3d.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name=f'{names[code]} (n = {len(sub)})',
            marker=dict(size=7, color=colors[code], opacity=0.88, line=dict(width=0.9, color='white')),
        ))

        mean = np.mean(pts, axis=1)
        cov = np.cov(pts)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        radii = 1.5 * np.sqrt(eigvals)

        u = np.linspace(0, 2 * np.pi, 36)
        v = np.linspace(0, np.pi, 18)
        xc = radii[0] * np.outer(np.cos(u), np.sin(v))
        yc = radii[1] * np.outer(np.sin(u), np.sin(v))
        zc = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        ellipsoid = np.einsum('ij,jkl->ikl', eigvecs, np.array([xc, yc, zc])) + mean[:, None, None]

        fig3d.add_trace(go.Surface(
            x=ellipsoid[0], y=ellipsoid[1], z=ellipsoid[2],
            opacity=0.15,
            colorscale=[[0, surf_rgba[code]], [1, surf_rgba[code]]],
            showscale=False, showlegend=False, hoverinfo='skip',
        ))

        pc1 = eigvecs[:, -1]
        if pc1[0] < 0:
            pc1 = -pc1
        t_val = np.dot(pts.T - mean, pc1)
        p_start = mean + np.min(t_val) * 1.1 * pc1
        p_end = mean + np.max(t_val) * 1.1 * pc1

        fig3d.add_trace(go.Scatter3d(
            x=[p_start[0], p_end[0]],
            y=[p_start[1], p_end[1]],
            z=[p_start[2], p_end[2]],
            mode='lines',
            line=dict(color=colors[code], width=7, dash='dash'),
            showlegend=False,
        ))

    fig3d.update_layout(
        font=dict(color=INK, size=16, family='Arial'),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            xaxis=dict(
                title=dict(text='Symptom duration (months)', font=dict(size=18, color=INK, family='Arial')),
                tickfont=dict(size=14, color=INK2),
                showbackground=False, gridcolor='#dddddd', zerolinecolor='#cccccc',
            ),
            yaxis=dict(
                title=dict(text='Treatment duration (months)', font=dict(size=18, color=INK, family='Arial')),
                tickfont=dict(size=14, color=INK2),
                showbackground=False, gridcolor='#dddddd', zerolinecolor='#cccccc',
            ),
            zaxis=dict(
                title=dict(text=LABELS[target], font=dict(size=18, color=INK, family='Arial')),
                tickfont=dict(size=14, color=INK2),
                range=[0, 10.5],
                showbackground=False, gridcolor='#dddddd', zerolinecolor='#cccccc',
            ),
            camera=dict(eye=dict(x=1.42, y=1.42, z=0.82)),
        ),
    )
    fig3d.write_html(os.path.join(OUT_DIR, html_name))
    return fig3d


fig_s6a = build_3d_space('I_VAS', '3D_Space_IVAS.html')
fig_s6b = build_3d_space('F_VAS', '3D_Space_FVAS.html')

img_a = trim_whitespace(Image.open(io.BytesIO(fig_s6a.to_image(format='png', width=1400, height=1100, scale=3))), pad=40)
img_b = trim_whitespace(Image.open(io.BytesIO(fig_s6b.to_image(format='png', width=1400, height=1100, scale=3))), pad=40)
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.46))
fig.subplots_adjust(wspace=0.06)
for ax, im, let in zip(axes, [img_a, img_b], ['a', 'b']):
    ax.imshow(im)
    ax.axis('off')
    add_panel(ax, let, x=0.0, y=1.02)
fig.legend(
    handles=[
        Line2D([], [], marker='o', ls='', color=NAVY, ms=5.5, label='Acute TMD (n = 51)'),
        Line2D([], [], marker='o', ls='', color=ROSE, ms=5.5, label='Chronic TMD (n = 38)'),
        Line2D([], [], color=INK2, lw=1.6, ls='--', label='Group trend line (1.5 SD ellipsoid shaded)'),
    ],
    loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=7,
)
save_all(fig, 'Supplementary_Figure_S6_3D_distributions')

print('✅ All figures (S2, S4, S5, S6) & HTMLs regenerated cleanly!')
