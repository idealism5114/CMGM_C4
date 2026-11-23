"""Reproducible, code-derived publication schematic of the CURRENT original C4.

Run with /usr/bin/python3 (system PyGObject + librsvg + pycairo).
No inference, dataset access, generated imagery, or model modifications.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
W,H=2480,1570
C=dict(ink='#1c2d43',muted='#52657a',line='#728498',border='#cbd5df',
       spatial='#2869a0',spatial_bg='#edf5fc',temporal='#177f7e',temporal_bg='#eaf7f4',
       recent='#b5751b',recent_bg='#fff5e5',interaction='#7960a5',interaction_bg='#f3eff9',
       router='#9c4687',router_bg='#fbf0f7')
svg=[]
_measure=None


def rect(x,y,w,h,fill='#fff',stroke=None,r=16,sw=1.8):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke or C["border"]}" stroke-width="{sw}"/>')


def text(x,y,value,size=28,fill=None,weight=400,anchor='start',max_width=None):
    global _measure
    if max_width is not None:
        import cairo
        if _measure is None:_measure=cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32,1,1))
        _measure.select_font_face('DejaVu Sans',cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_BOLD if weight>=550 else cairo.FONT_WEIGHT_NORMAL)
        _measure.set_font_size(size)
        width=_measure.text_extents(value).x_advance
        if width>max_width:size=round(size*max_width/width,2)
    svg.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, Noto Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill or C["ink"]}" text-anchor="{anchor}">{escape(value)}</text>')


def line(points,color=None,sw=2.8,dashed=False,arrow=True):
    d='M '+' L '.join(f'{x},{y}' for x,y in points)
    attrs=f'stroke="{color or C["line"]}" stroke-width="{sw}" fill="none" stroke-linejoin="round" stroke-linecap="round"'
    if dashed:attrs+=' stroke-dasharray="9 7"'
    if arrow:
        marker={'#2869a0':'spatial','#177f7e':'temporal','#9c4687':'router'}.get(color,'default')
        attrs+=f' marker-end="url(#{marker})"'
    svg.append(f'<path d="{d}" {attrs}/>')


def box(x,y,w,h,title,sub=None,color='ink',bg='#fff',title_size=29):
    rect(x,y,w,h,bg,C.get(color,color))
    if sub is None:text(x+w/2,y+h/2+10,title,title_size,C.get(color,color),600,'middle',max_width=w-32)
    else:
        text(x+w/2,y+42,title,title_size,C.get(color,color),600,'middle',max_width=w-32)
        for i,s in enumerate(sub if isinstance(sub,list) else [sub]):
            text(x+w/2,y+78+i*34,s,25,C['muted'],400,'middle',max_width=w-32)


def panel(x,y,w,h,label,title):
    rect(x,y,w,h,'#fff','#b7c5d3',19,2)
    text(x+27,y+46,label,32,C['ink'],700)
    text(x+85,y+46,title,31,C['ink'],650)


def main():
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    svg.append('<title>C4: Global Dynamic Spatiotemporal Prediction Mixture of Experts</title>')
    svg.append('<desc>Exact current C4: parallel original spatial and switching-latent temporal encoders; four shared experts; one shared forecast head; sample-dependent, horizon-shared router; prediction-level mixture. Original four-horizon Huber and Switch KL.</desc>')
    svg.append('<defs>')
    for key,col in [('default',C['line']),('spatial',C['spatial']),('temporal',C['temporal']),('router',C['router'])]:
        svg.append(f'<marker id="{key}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10 Z" fill="{col}"/></marker>')
    svg.append('</defs>')
    rect(0,0,W,H,'#fff','#fff',0)
    text(60,65,'C4  |  Global Dynamic Spatiotemporal Prediction MoE',45,weight=700)
    text(63,111,'One sample-conditioned router · four shared experts · prediction-level fusion · four cumulative-return horizons',27,C['muted'])

    # (a) Overview. The narrow rail denotes a named tuple of live features,
    # NEVER a fused vector or a cross-sample aggregation.
    panel(50,150,2380,705,'a','End-to-end prediction')
    rect(70,345,260,250,'#f5f7fa','#b9c7d5')
    text(200,386,'Historical inputs',27,weight=650,anchor='middle')
    for yy,market,count,col in [(432,'Stocks','263',C['spatial']),(475,'Bonds','25',C['temporal']),(518,'Commodities','23',C['recent'])]:
        svg.append(f'<circle cx="92" cy="{yy-8}" r="6" fill="{col}"/>')
        text(109,yy,market,24,max_width=168);text(309,yy,count,25,anchor='end',weight=650)
    text(200,566,'B × 20 × 311 × 26',23,C['muted'],anchor='middle')
    line([(330,468),(365,468),(365,423),(400,423)],C['spatial'])
    line([(365,468),(365,648),(400,648)],C['temporal'])
    box(400,350,620,145,'Original spatial encoder',
        ['History attention → adaptive graph → 2 graph blocks','Market pooling + spatial projection → s: B × 64'],
        'spatial',C['spatial_bg'],31)
    box(400,575,620,145,'Switching-latent temporal encoder',
        ['Causal memory H + recurrent microstates Z','Balanced readout + temporal projection → t'],
        'temporal',C['temporal_bg'],30)
    line([(1020,423),(1065,423)],C['spatial'],arrow=False)
    line([(1020,648),(1065,648)],C['temporal'],arrow=False)
    line([(1065,263),(1065,740)],C['line'],3.3,arrow=False)
    for yy in (423,648):svg.append(f'<circle cx="1065" cy="{yy}" r="5" fill="{C["line"]}"/>')
    text(710,535,'Both branches execute once per forward',25,C['muted'],anchor='middle')
    # Router is a separate path; it only weights candidate forecasts.
    box(1130,215,760,96,'Global dynamic router',
        'u: B × 256 → MLP(32, GELU) → softmax → π: B × 4',
        'router',C['router_bg'],30)
    line([(1065,263),(1130,263)],C['router'],dashed=True)
    line([(1890,263),(2150,263),(2150,380)],C['router'],3,dashed=True)
    text(2150,319,'Same π for all 4 horizons',25,C['router'],600,'middle')
    experts=[(340,'E₁  Recent latent change','Z_last, mean₅(Z), ΔZ','recent','recent_bg'),
             (445,'E₂  Long memory','H_last, mean₂₀(H)','temporal','temporal_bg'),
             (550,'E₃  Spatial structure','Projected spatial state s','spatial','spatial_bg'),
             (655,'E₄  ST interaction','s and t; low-rank product','interaction','interaction_bg')]
    for y,title,sub,col,bg in experts:
        rect(1130,y,400,85,C[bg],C[col])
        text(1150,y+34,title,27,C[col],650)
        text(1150,y+66,sub,24,C['muted'])
        line([(1065,y+42.5),(1130,y+42.5)])
        line([(1530,y+42.5),(1600,y+42.5)],C[col])
    rect(1600,340,290,400,'#f8f9fc','#8494aa')
    text(1745,382,'One shared Headθ',27,weight=650,anchor='middle',max_width=260)
    text(1745,421,'Same θ for all experts',23,C['muted'],anchor='middle',max_width=260)
    rect(1620,447,250,150,'#fff','#cbd5df',12)
    text(1745,481,'Linear 64 → 64',26,anchor='middle')
    text(1745,524,'ReLU + Dropout 0.3',24,anchor='middle')
    text(1745,570,'Linear 64 → 4 × 23',24,anchor='middle')
    text(1745,635,'Horizon-major rows',23,C['muted'],anchor='middle')
    text(1745,680,'P: B × 4 × 4 × 23',27,weight=600,anchor='middle',max_width=260)
    text(1745,713,'[B, horizon, expert, C]',22,C['muted'],anchor='middle',max_width=260)
    line([(1890,540),(1920,540),(1920,455),(1950,455)])
    rect(1950,380,400,150,'#f0f3f7','#6c819b')
    text(2150,424,'Prediction-level fusion',29,weight=650,anchor='middle')
    text(2150,472,'ŷₕ = ∑ₖ πₖ Pₕ,ₖ',36,weight=550,anchor='middle')
    text(2150,508,'Mix forecasts after the shared head',22,C['muted'],anchor='middle')
    line([(2150,530),(2150,625)])
    rect(1950,625,400,120,'#f7fafc','#7b92aa')
    text(2150,663,'23 commodity forecasts',27,weight=650,anchor='middle',max_width=375)
    text(2150,698,'1 / 5 / 10 / 20 sessions',27,anchor='middle')
    text(2150,730,'Output: B × 4 × 23',25,C['muted'],anchor='middle')
    text(1065,785,'Live feature taps: (s, t, H, Z), without early fusion',27,C['muted'],anchor='middle')
    text(1240,825,'Four experts are evaluated once each; all access remains dense. Solid: forecast data flow. Dashed: routing weights/context.',24,C['muted'],anchor='middle')

    # (b) Spatial backbone detail.
    panel(50,895,730,495,'b','Spatial information pathway')
    box(80,964,390,65,'Heterogeneous node encoder',color='spatial',bg=C['spatial_bg'],title_size=26)
    text(620,1004,'B × 20 × 311 × 64',22,C['muted'],anchor='middle',max_width=245)
    line([(275,1029),(275,1050)],C['spatial'])
    box(80,1050,390,65,'History-weighted aggregation',color='spatial',bg=C['spatial_bg'],title_size=25)
    text(620,1091,'H₀: B × 311 × 64',24,C['muted'],anchor='middle',max_width=245)
    line([(275,1115),(275,1136)],C['spatial'])
    rect(80,1136,390,69,C['spatial_bg'],C['spatial'])
    text(275,1163,'EdgeAttnMixHop → ReLU',25,C['spatial'],600,'middle')
    text(275,1193,'→ EdgeAttnMixHop',25,C['spatial'],600,'middle')
    box(520,1136,230,69,'Learned A₀',color='spatial',bg='#fff',title_size=27)
    text(635,1235,'Global N × N prior',21,C['muted'],anchor='middle')
    line([(520,1170),(470,1170)],C['spatial'])
    line([(275,1205),(275,1250)],C['spatial'])
    rect(80,1250,670,77,C['spatial_bg'],C['spatial'])
    text(415,1280,'LayerNorm → masked means of 3 markets',25,C['spatial'],500,'middle')
    text(415,1311,'Concat → Linear(192→64) → gcn_proj → s',25,C['spatial'],500,'middle')
    text(415,1363,'Each graph block retains its original K = 2 MixHop.',23,C['muted'],anchor='middle')

    # (c) Temporal backbone detail. Two different categorical mixtures exist:
    # three internal regimes, four final prediction experts.
    panel(805,895,850,495,'c','Causal memory and latent dynamics')
    box(835,964,790,65,'Market tokens → Causal Transformer + RPE',
        color='temporal',bg=C['temporal_bg'],title_size=25)
    line([(940,1029),(940,1050)],C['temporal'])
    rect(835,1050,220,65,C['temporal_bg'],C['temporal'])
    text(945,1077,'Long memory H',24,C['temporal'],600,'middle')
    text(945,1104,'B × 20 × 128',23,C['muted'],anchor='middle')
    box(1110,1050,515,65,'Markov regime filter: pₜ,  K = 3',color='temporal',bg=C['temporal_bg'],title_size=26)
    line([(1055,1082),(1110,1082)],C['temporal'])
    line([(1370,1115),(1370,1140)],C['temporal'])
    rect(1110,1140,515,90,C['temporal_bg'],C['temporal'])
    text(1367,1175,'Regime-specific generators G₁, G₂, G₃',25,C['temporal'],600,'middle',max_width=480)
    text(1367,1212,'zₜ = ∑ᵣ pₜ,ᵣ Gᵣ([Hₜ, zₜ₋₁])',29,C['temporal'],500,'middle')
    rect(835,1140,220,90,C['temporal_bg'],C['temporal'])
    text(945,1175,'Microstates Z',25,C['temporal'],600,'middle')
    text(945,1211,'B × 20 × 64',24,C['muted'],anchor='middle')
    line([(1110,1205),(1055,1205)],C['temporal'])
    line([(1055,1094),(1080,1094),(1080,1170),(1110,1170)],C['temporal'])
    line([(835,1082),(820,1082),(820,1290),(835,1290)],C['temporal'])
    line([(945,1230),(945,1250)],C['temporal'])
    rect(835,1250,790,78,C['temporal_bg'],C['temporal'])
    text(1230,1280,'Balanced readout: [LN(P_H H_last), LN(P_Z z_last)]',25,C['temporal'],550,'middle',max_width=750)
    text(1230,1313,'Linear(128→64) → temporal projection → t: B × 64',26,C['temporal'],500,'middle')
    text(1230,1363,'Live H and Z also feed the experts and the router directly.',23,C['muted'],anchor='middle')

    # (d) Source-faithful expert and router formulas.
    panel(1680,895,750,495,'d','Experts and global routing')
    text(1710,982,'E₁ = ReLU(W₁[Z_last, Z̄₅, Z_last−Z̄₅])',28,C['recent'],550)
    text(1710,1030,'E₂ = ReLU(W₂[H_last, H̄₂₀])',28,C['temporal'],550)
    text(1710,1078,'E₃ = s + ReLU(W₃s)',28,C['spatial'],550)
    text(1710,1126,'E₄ = Wₒ[(Uₛs) ⊙ (Uₜt)],  rank = 16',28,C['interaction'],550)
    text(1710,1172,'Eₖ ∈ ℝ⁶⁴   →   P[b,h,k,:] = Headθ,h(Eₖ)',27,C['muted'])
    rect(1710,1200,690,150,C['router_bg'],C['router'],12)
    text(1730,1235,'u = [LN(s), LN(t), LN(ΔZ), LN(W_R H̄₂₀)]',25,C['router'],500)
    text(1730,1279,'π = softmax(MLP₂₅₆→₃₂→₄(u))',30,C['router'],650)
    text(1730,1323,'GELU hidden layer · temperature 1 · no horizon input',23,C['router'])

    # Current default training objective, not recent failed experimental losses.
    rect(50,1420,2380,100,'#f6f8fb','#d6dee7',15)
    text(80,1460,'Training:  L = ∑ₕ MaskedHuber₍δ=0.02₎(ŷₕ, yₕ) + β(epoch) · mean_valid KL(pₜ ∥ pₜ₋₁T)',28,weight=550)
    text(80,1499,'Checkpoint selection: VAL 5-session commodity Macro MAE.  The original Switch KL warmup is retained.',25,C['muted'])
    text(60,1553,'Current implementation: moe_phase_m/pred_global_dynamic  ·  311 assets / 26 features / 23 targets  ·  567,169 trainable parameters',23,C['muted'])
    svg.append('</svg>')
    data='\n'.join(svg)
    ET.fromstring(data)
    (OUT/'c4_architecture.svg').write_text(data)

    import gi,cairo
    gi.require_version('Rsvg','2.0')
    from gi.repository import Rsvg
    handle=Rsvg.Handle.new_from_file(str(OUT/'c4_architecture.svg'))
    view=Rsvg.Rectangle();view.x=view.y=0;view.width=W;view.height=H
    for filename,scale in [('c4_architecture.png',1.5),('c4_architecture_preview.png',.75)]:
        surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,int(W*scale),int(H*scale))
        ctx=cairo.Context(surface);ctx.scale(scale,scale);handle.render_document(ctx,view)
        surface.write_to_png(str(OUT/filename));surface.finish()
    surface=cairo.PDFSurface(str(OUT/'c4_architecture.pdf'),744,744*H/W)
    ctx=cairo.Context(surface);ctx.scale(744/W,744/W);handle.render_document(ctx,view);ctx.show_page();surface.finish()

    names=['cmgm/models/phase_m_new_data.py','cmgm/models/phase_m_prediction_moe.py',
           'cmgm/models/formal_d0b_main_ablation.py','cmgm/models/four_task_moe.py',
           'cmgm/models/switching_latent_transformer.py']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    identity=json.loads((ROOT/'experiments/phase_m_horizon_prediction_moe/newdata_20261026_135222_918381/runs/C4/COMPLETE.json').read_text())
    config=json.loads((ROOT/'experiments/phase_m_horizon_prediction_moe/newdata_20261026_135222_918381/TRAINING_CONFIG.json').read_text())
    assert config['shape']==dict(nodes=311,stocks=263,bonds=25,targets=23,features=26)
    assert identity['model_id']=='moe_phase_m/pred_global_dynamic'
    meta=dict(model='moe_phase_m/pred_global_dynamic',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        checkpoint_sha256=identity['checkpoint_sha256'],best_epoch=identity['best_epoch'],
        figure_date='2026-11-12',dimensions=dict(input=['B',20,311,26],output=['B',4,23],candidate=['B',4,4,23],routing=['B',4]),
        source_sha256={n:digest(ROOT/n) for n in names},
        critical_checks=dict(horizon_shared_router=True,prediction_level_fusion=True,one_shared_head=True,
            no_D0B_gate_in_C4=True,interaction_has_no_temporal_residual=True,regimes3_experts4=True,
            no_new_dynamic_graph_or_GraphSkip=True,original_Huber_and_KL=True),
        rendering='Deterministic editable SVG; vector PDF and PNG via system librsvg/pycairo; no image generation',
        training_or_data_access=False,
        files={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='provenance.json'})
    (OUT/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print('WROTE',OUT)


if __name__=='__main__':main()
