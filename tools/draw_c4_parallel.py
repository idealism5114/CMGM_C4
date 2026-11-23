"""Icon-led, source-faithful C4 architecture. SVG/PDF/PNG; no model/data access.

Run: /usr/bin/python3 tools/draw_c4_parallel.py
Requires system librsvg, PyGObject and pycairo; no packages are installed.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/c4_architecture_parallel_20261118'
W, H = 3300, 1500
C = dict(ink='#24354b', muted='#627387', line='#8290a0', border='#d6e0ea',
         blue='#397fb5', blue_bg='#f1f7fc', teal='#2a918d', teal_bg='#eff8f5',
         orange='#d79436', purple='#8d72b5', pink='#b65a8b', pink_bg='#fbf1f6',
         macro='#67a795', micro='#84a04c', stock='#5c90bc', bond='#63a98e', commodity='#d4a354')
E_COLORS = [C['orange'], C['teal'], C['blue'], C['purple']]
S = []
LABELS = {
    'en': dict(title='C4 · Global Dynamic Spatiotemporal Prediction MoE',
        input='History',stocks='Stocks',bonds='Bonds',commodities='Commodities',
        spatial='Spatial branch',temporal='Temporal branch',node='Node encoder',history='History attention',
        graph='Adaptive graph',pool='Market pooling',projection='Projection',market='Market encoder',
        macro='Macro memory',transformer='Causal Transformer',micro='Micro dynamics',regime='Regime filter',
        generator='Regime-specific generators',latent='Latent recurrence',balance='Balanced readout',
        states='Live states',router='Global dynamic router',shared='Shared head',one='same θ for all experts',
        candidates='Candidate forecasts',prediction='Forecasts',recent='Recent change',
        memory='Long memory',structure='Spatial structure',interaction='ST interaction',
        mixture='Prediction fusion',same='One π per sample · shared by all four horizons',
        feature='Representation / prediction',control='Prior / routing weights',
        input_shape='20 × 311 × 26',output_shape='B × 4 × 23',moe='Prediction-level MoE',
        readout='Project + LN',sources='Sources: s, t, ΔZ, mean(H)',periods='1 / 5 / 10 / 20 sessions',
        caption='Macro memory drives microstate updates; H and Z also feed the experts directly.'),
    'zh': dict(title='C4 · 全局动态时空预测混合专家模型',
        input='历史市场输入',stocks='股票',bonds='债券',commodities='商品',spatial='空间分支',temporal='时间分支',
        node='异构节点编码',history='历史时间加权',graph='自适应图',pool='分市场池化',projection='投影',
        market='市场编码',macro='宏观记忆',transformer='因果 Transformer',micro='微观动态',regime='状态识别',
        generator='状态专属生成器',latent='潜在状态递推',balance='平衡读出',states='实时表示',
        router='全局动态路由',shared='共享预测头',one='四专家共用同一组 θ',candidates='专家候选预测',
        prediction='收益率预测',recent='近期状态变化',memory='长期记忆',structure='空间结构',interaction='时空交互',
        mixture='预测结果融合',same='每样本一组 π · 四个预测期限共用',feature='表示／预测数据流',control='图先验／路由权重',
        input_shape='20 × 311 × 26',output_shape='B × 4 × 23',moe='预测结果层 MoE',readout='投影 + LN',
        sources='输入：s、t、ΔZ、mean(H)',periods='1 / 5 / 10 / 20 个索引步',
        caption='宏观记忆驱动微观状态更新；H、Z 同时直接供给专家。')
}


def rect(x,y,w,h,fill='white',stroke=None,r=13,sw=1.8,opacity=1):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{sw}" opacity="{opacity}"/>')


def txt(x,y,value,size=27,color=None,weight=400,anchor='middle',width=None):
    attrs = f' textLength="{width}" lengthAdjust="spacingAndGlyphs"' if width else ''
    S.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color or C["ink"]}" text-anchor="{anchor}"{attrs}>{escape(str(value))}</text>')


def path(points,color=None,sw=3,dashed=False,arrow=True):
    col = color or C['line']
    d='M '+' L '.join(f'{x},{y}' for x,y in points)
    extra=' stroke-dasharray="9 7"' if dashed else ''
    if arrow:extra+=f' marker-end="url(#{"a"+col[1:]})"'
    S.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"{extra}/>')


def circle(x,y,r,fill='white',stroke=None,sw=2):
    S.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{sw}"/>')


def icon(name,x,y,col,scale=1):
    S.append(f'<g transform="translate({x},{y}) scale({scale})" stroke-linecap="round" stroke-linejoin="round">')
    if name=='stock':
        for xx,hh in [(-37,34),(-10,57),(17,76)]:
            rect(xx,34-hh,19,hh,'white',col,2,2.5)
            for yy in range(int(42-hh),27,14):path([(xx+6,yy),(xx+13,yy)],col,2,arrow=False)
        path([(-44,44),(44,44)],col,3,arrow=False)
    elif name=='bond':
        rect(-39,-44,78,89,'white',col,6,2.5)
        for yy,w in [(-20,49),(-3,42),(15,29)]:path([(-24,yy),(-24+w,yy)],col,2.5,arrow=False)
        circle(21,31,12,'white',col,2.5)
        path([(15,41),(11,54),(22,49),(30,54),(28,41)],col,2.5,arrow=False)
    elif name=='commodity':
        for yy in [-23,6,35]:
            S.append(f'<path d="M-37,{yy+10} L-25,{yy-10} L21,{yy-10} L37,{yy+10} Z" fill="white" stroke="{col}" stroke-width="2.5"/>')
    elif name in ('encoder','market'):
        for k,cc in enumerate([C['stock'],C['bond'],C['commodity']]):
            for j in range(4):rect(-53+j*13,-40+k*28,9,17,cc,cc,2)
        for yy in [-31,-3,25]:path([(6,yy),(26,yy)],col,2,arrow=False)
        for k,cc in enumerate([C['stock'],C['bond'],C['commodity']]):rect(30,-40+k*28,32,18,'white',cc,3,2.3)
        if name=='market':
            path([(47,50),(-42,50)],col,2,arrow=False)
            for k in range(4):rect(-44+k*26,42,18,16,col,col,2)
    elif name=='attention':
        for j in range(5):
            h=[22,36,58,42,26][j]
            rect(-55+j*25,30-h,15,h,col,col,2,opacity=[.25,.5,1,.65,.35][j])
        path([(-57,37),(61,37)],col,2,arrow=False)
        path([(0,48),(0,64)],col,2.8)
        rect(-28,72,56,12,col,col,3)
    elif name=='graph':
        pts=[(-46,-8),(-19,-37),(17,-25),(47,0),(17,37),(-27,35),(0,2)]
        for a,b in [(0,1),(0,5),(0,6),(1,2),(1,6),(2,3),(2,6),(3,4),(3,6),(4,5),(4,6),(5,6)]:
            path([pts[a],pts[b]],col,2,arrow=False)
        for j,(xx,yy) in enumerate(pts):circle(xx,yy,8,[C['stock'],C['bond'],C['commodity']][j%3],col,1.5)
    elif name=='adjacency':
        for i in range(5):
            for j in range(5):rect(-37+j*16,-37+i*16,12,12,col,col,1,opacity=[.12,.35,.6,.9][(i*3+j*7)%4])
    elif name=='pool':
        for k,cc in enumerate([C['stock'],C['bond'],C['commodity']]):
            xx=-47+k*45
            for q in range(3):circle(xx+(q-1)*10,-28+abs(q-1)*9,5,cc)
            path([(xx,-8),(xx,8)],cc,2.1)
            rect(xx-12,14,24,24,cc,cc,4)
        path([(-47,53),(0,73),(43,53)],col,2,arrow=False)
        rect(-22,79,44,14,col,col,3)
    elif name=='project':
        for j in range(5):rect(-41+j*17,-39,11,28,col,col,2,opacity=.35+.13*j)
        path([(0,0),(0,24)],col,2.8)
        for j in range(3):rect(-25+j*21,36,14,31,col,col,2)
    elif name=='transformer':
        for q in range(3):
            xx=-51+q*40
            rect(xx,-44,30,90,'white',col,5,2.3)
            for i in range(4):
                for j in range(4):
                    rect(xx+4+j*5.5,-26+i*12,4.2,7,col,col,0,opacity=.6 if j<=i else .08)
        path([(-46,62),(56,62)],col,2.2)
    elif name=='memory':
        for q in range(3):
            rect(-39+q*13,-39-q*9,58,76,'white',col,5,2.5)
        for yy in [-30,-12,6]:path([(-3,yy),(29,yy)],col,2.7,arrow=False)
    elif name=='regime':
        pts=[(-34,22),(0,-33),(38,22)]
        for a,b in [(0,1),(1,2),(2,0)]:path([pts[a],pts[b]],col,2.3,arrow=False)
        for k,(xx,yy) in enumerate(pts):circle(xx,yy,14,[C['macro'],C['micro'],C['teal']][k],col,1.5)
    elif name=='generators':
        for q,cc in enumerate([C['macro'],C['micro'],C['teal']]):
            xx=-49+q*36
            rect(xx,-44,27,86,'white',cc,5,2.1)
            for yy in [-25,-1,23]:circle(xx+13,yy,4.5,cc)
            path([(xx+13,-19),(xx+13,-7)],cc,1.6,arrow=False)
            path([(xx+13,5),(xx+13,17)],cc,1.6,arrow=False)
    elif name=='latent':
        for j in range(3):
            rect(-54+j*42,-23,29,48,'white',col,7,2.5)
            if j<2:path([(-25+j*42,0),(-14+j*42,0)],col,2)
            for k in range(3):path([(-47+j*42,-11+k*11),(-33+j*42,-11+k*11)],col,1.9,arrow=False)
    elif name=='balance':
        for yy,cc in [(-35,C['macro']),(28,C['micro'])]:
            rect(-55,yy-15,43,28,'white',cc,4,2)
            txt(-33,yy+5,'LN',15,cc,600)
            path([(-6,yy),(20,0)],cc,2.3,arrow=False)
        rect(23,-25,27,50,col,col,4)
    elif name=='recent':
        path([(-55,25),(-37,17),(-19,29),(-3,-3),(13,11),(34,-28),(54,-35)],col,3.5,arrow=False)
        circle(54,-35,5,col)
        path([(28,33),(52,7)],col,2.2)
    elif name=='interaction':
        for xx,cc in [(-38,C['blue']),(37,C['teal'])]:
            rect(xx-12,-43,24,32,cc,cc,4)
            path([(xx,-4),(0,12)],cc,2.5,arrow=False)
        circle(0,22,17,'white',col,2.5)
        path([(-7,15),(7,29)],col,2.3,arrow=False)
        path([(-7,29),(7,15)],col,2.3,arrow=False)
        path([(0,43),(0,61)],col,2.5)
    elif name=='mlp':
        layers=[(-46,[-32,0,32]),(0,[-48,-16,16,48]),(46,[-32,0,32])]
        for (xa,ya),(xb,yb) in zip(layers,layers[1:]):
            for aa in ya:
                for bb in yb:path([(xa,aa),(xb,bb)],col,1.2,arrow=False)
        for xx,ys in layers:
            for yy in ys:circle(xx,yy,6,'white',col,2)
    S.append('</g>')


def panel(x,y,w,h,label,col,bg):
    rect(x,y,w,h,bg,C['border'],20,2)
    rect(x+19,y+22,7,34,col,col,3)
    txt(x+44,y+49,label,32,col,650,'start')


def label(x,y,text,col,size=26):txt(x,y,text,size,col,550)


def card(x,y,w,h,col,fill='white'):
    rect(x,y,w,h,fill,col,13,1.9)


def mini_forecast(x,y,col,rows=4,cols=7,cell=13):
    for i in range(rows):
        for j in range(cols):
            opacity=.17+.68*((i*3+j*5)%11)/10
            rect(x+j*cell,y+i*cell,cell-3,cell-3,col,col,1,opacity=opacity)


def draw(lang):
    global S
    S=[];L=LABELS[lang]
    S.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    S.append('<title>Current C4: parallel spatial and macro–micro temporal encoding with prediction-level MoE</title>')
    S.append('<desc>One global dynamic four-expert router per sample, shared across horizons; one shared forecast head. Macro memory H drives regime filtering and recurrent microstates Z. Learned global A0 feeds two original graph blocks. No D0B gate, no task router, no new graph mechanism.</desc>')
    S.append('<defs>')
    for col in sorted(set(C.values())|set(E_COLORS)):
        S.append(f'<marker id="a{col[1:]}" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L11,5.5 L0,11 Z" fill="{col}"/></marker>')
    S.append('</defs>')
    rect(0,0,W,H,'white','white',0)
    txt(40,70,L['title'],43,weight=650,anchor='start')
    # Input: schematic icons, exact current dimensions and fixed market counts.
    panel(30,485,220,565,'',C['ink'],'#fafbfd')
    txt(74,534,L['input'],26,C['ink'],650,'start')
    for y,name,count,key in [(625,'stock',263,'stocks'),(780,'bond',25,'bonds'),(935,'commodity',23,'commodities')]:
        icon(name,84,y-35,C[name],.8)
        txt(179,y-39,count,31,C['ink'],650)
        txt(179,y-6,L[key],19,C['muted'])
    txt(140,1012,L['input_shape'],24,C['muted'])
    path([(250,766),(275,766),(275,411),(324,411)],C['blue'],3.5)
    path([(275,766),(275,911),(326,911)],C['teal'],3.5)
    circle(275,766,5,C['line'])

    # Parallel original spatial encoder. A0 is learned globally, not conditioned on this sample.
    panel(305,145,1510,415,L['spatial'],C['blue'],C['blue_bg'])
    modules=[(423,'encoder','node'),(657,'attention','history'),(942,'graph',None),(1177,'graph',None),(1422,'pool','pool'),(1680,'project','projection')]
    for x,name,key in modules:
        icon(name,x,390,C['blue'])
        if key:label(x,505,L[key],C['blue'],25)
    for a,b in [(492,588),(723,863),(1008,1102),(1248,1343),(1491,1605)]:path([(a,411),(b,411)],C['blue'])
    txt(942,500,'EdgeAttn',25,C['blue'],600);txt(942,529,'MixHop ①',24,C['blue'])
    txt(1177,500,'EdgeAttn',25,C['blue'],600);txt(1177,529,'MixHop ②',24,C['blue'])
    # A0 prior has a single source and two consumers.
    icon('adjacency',1038,246,C['blue'],.66)
    txt(1090,242,L['graph'],24,C['blue'],550,'start')
    txt(1090,272,'A₀',24,C['muted'],500,'start')
    path([(1038,278),(1038,303),(942,303),(942,337)],C['blue'],2.5,True)
    path([(1038,303),(1177,303),(1177,337)],C['blue'],2.5,True)
    txt(1036,541,'ReLU',19,C['muted'])
    txt(1422,537,'LN · mean · concat',19,C['muted'])
    path([(1728,411),(1878,411)],C['blue'])
    txt(1774,387,'s',29,C['blue'],650)
    txt(1680,537,'64 → 64',20,C['muted'])

    # Macro / micro are visually separate lanes, with explicit H -> micro links.
    panel(305,600,1510,755,L['temporal'],C['teal'],C['teal_bg'])
    rect(610,681,670,238,'#f7fbf9','#c5dfd5',15,1.5)
    txt(633,720,L['macro'],29,C['teal'],650,'start')
    rect(610,976,670,303,'#f9fbf4','#d5dfbf',15,1.5)
    txt(633,1016,L['micro'],29,C['micro'],650,'start')
    icon('market',423,885,C['teal'])
    label(423,1004,L['market'],C['teal'],25)
    path([(492,911),(550,911),(550,805),(680,805)],C['teal'])
    icon('transformer',807,805,C['teal'],1.06)
    label(807,889,L['transformer'],C['teal'],24)
    path([(878,805),(1040,805)],C['teal'])
    icon('memory',1123,805,C['teal'])
    txt(1123,891,'H · 20 × 128',24,C['teal'],600)
    # Micro: H_t + previous regime/state; three distinct generators, not the four MoE experts.
    icon('regime',722,1115,C['micro'])
    icon('generators',929,1115,C['teal'])
    icon('latent',1154,1115,C['micro'])
    label(722,1201,L['regime'],C['micro'],23)
    txt(722,1235,'K = 3',20,C['muted'])
    if lang=='en':
        label(929,1193,'Regime-specific',C['teal'],22)
        label(929,1221,'generators',C['teal'],22)
    else:label(929,1201,L['generator'],C['teal'],21)
    txt(1154,1201,'Z · 20 × 64',23,C['micro'],600)
    path([(766,1115),(863,1115)],C['micro'])
    txt(815,1097,'pₜ',23,C['micro'])
    path([(988,1115),(1089,1115)],C['teal'])
    path([(1123,852),(1123,944),(722,944),(722,1066)],C['teal'],2.7)
    path([(929,944),(929,1064)],C['teal'],2.7)
    txt(1027,934,'Hₜ',22,C['teal'])
    path([(1217,1115),(1250,1115),(1250,1257),(1046,1257),(1046,1154),(990,1154)],C['micro'],2.5)
    txt(1140,1246,'Zₜ₋₁',21,C['micro'])
    # H_last / Z_last balanced readout, then original temporal projection.
    card(1360,872,210,206,C['teal'],'white')
    icon('balance',1465,952,C['teal'],1.03)
    txt(1465,1044,L['readout'],21,C['teal'],550)
    txt(1465,1115,L['balance'],25,C['teal'],600)
    path([(1188,805),(1305,805),(1305,920),(1360,920)],C['teal'])
    path([(1217,1115),(1305,1115),(1305,1010),(1360,1010)],C['micro'])
    txt(1327,900,'Hₜ',20,C['teal'])
    txt(1327,1045,'Zₜ',20,C['micro'])
    path([(1570,971),(1615,971)],C['teal'])
    icon('project',1679,950,C['teal'],.76)
    txt(1679,1060,'128 → 64 → 64',20,C['muted'])
    path([(1722,971),(1878,971)],C['teal'])
    txt(1776,949,'t',29,C['teal'],650)
    # H / Z bypass the readout to remain directly available to experts/router.
    path([(1261,805),(1261,663),(1860,663),(1860,710),(1878,710)],C['teal'],2.7)
    path([(1220,1115),(1290,1115),(1290,1320),(1860,1320),(1860,1250),(1878,1250)],C['micro'],2.7)
    txt(1838,694,'H',25,C['teal'],600)
    txt(1838,1292,'Z',25,C['micro'],600)
    # Named bundle, not an early fusion operation.
    path([(1880,279),(1880,1320)],C['line'],4,arrow=False)
    for y in [411,710,971,1250]:circle(1880,y,5.5,C['line'])
    txt(1880,228,L['states'],24,C['muted'],550)
    txt(1880,1352,'s, t, H, Z',22,C['muted'])

    # All expert, head, router and forecast-mixture operations belong to ONE MoE.
    # The final horizon/commodity output remains outside this common boundary.
    S.append('<g id="prediction-level-moe">')
    S.append('<g id="moe-boundary">')
    panel(1958,115,1007,1290,L['moe'],C['ink'],'#faf9fd')
    S.append('</g>')
    # A single sample-conditioned router, nested inside the MoE.
    S.append('<g id="moe-router" transform="translate(0,65)">')
    panel(1990,145,750,224,L['router'],C['pink'],C['pink_bg'])
    icon('mlp',2080,266,C['pink'],.65)
    path([(1880,279),(2007,279)],C['pink'],2.8,True)
    txt(2211,264,'MLP',27,C['pink'],600)
    txt(2211,299,'256 → 32 → 4',22,C['muted'])
    path([(2287,275),(2320,275)],C['pink'],2.6)
    txt(2373,282,'softmax',23,C['pink'],550)
    path([(2430,275),(2464,275)],C['pink'],2.6)
    for j,(cc,hh) in enumerate(zip(E_COLORS,[46,29,35,44])):
        rect(2490+j*45,300-hh,27,hh,cc,cc,3)
    txt(2582,329,'π(x)',25,C['pink'],600)
    path([(2660,275),(2726,275)],C['pink'],2.6,arrow=False)
    txt(2365,401,L['same'],25,C['pink'],550)
    S.append('</g>')

    # Four experts, one shared prediction head, then mixture of forecasts.
    S.append('<g id="moe-experts">')
    ys=[604,815,1026,1237]
    for j,(yy,name,key,source) in enumerate(zip(ys,['recent','memory','graph','interaction'],['recent','memory','structure','interaction'],['Z','H','s','s, t'])):
        cc=E_COLORS[j]
        path([(1880,yy),(1998,yy)],cc,2.8)
        txt(1936,yy-17,source,23,cc,550)
        card(2000,yy-77,290,156,cc,'white')
        icon(name,2060,yy-4,cc,.68)
        txt(2220,yy-17,f'E{j+1}',26,cc,650)
        txt(2145,yy+56,L[key],25,cc,600)
        txt(2220,yy+15,'64',20,C['muted'])
        path([(2290,yy),(2340,yy)],cc,2.6)
    S.append('</g>')
    # Same theta; channel schematic is not four independent prediction heads.
    S.append('<g id="moe-shared-head">')
    card(2340,526,190,788,C['line'],'#f4f6fa')
    txt(2435,567,L['shared'],27,C['ink'],650)
    icon('mlp',2435,684,C['muted'],.92)
    txt(2435,782,'θ',39,C['ink'],650)
    path([(2371,817),(2499,817)],C['line'],1.8,arrow=False)
    txt(2435,875,'Linear',24,C['ink'],550)
    txt(2435,911,'ReLU',24,C['ink'],550)
    txt(2435,947,'Dropout',24,C['ink'],550)
    txt(2435,995,'Linear',24,C['ink'],550)
    txt(2435,1062,'64 → 64',23,C['muted'])
    txt(2435,1096,'→ 4 × 23',23,C['muted'])
    txt(2435,1300,'one θ',24,C['ink'],600)
    S.append('</g>')
    S.append('<g id="moe-candidate-forecasts">')
    txt(2651,514,L['candidates'],23,C['muted'],550)
    for j,yy in enumerate(ys):
        cc=E_COLORS[j]
        path([(2530,yy),(2570,yy)],cc,2.6)
        rect(2574,yy-49,158,98,'white',C['border'],8,1.5)
        mini_forecast(2589,yy-34,cc,cell=17)
        txt(2712,yy+32,f'P{j+1}',18,cc,600)
        path([(2732,yy),(2804,yy)],cc,2.7,arrow=False)
    path([(2804,604),(2804,1237)],C['line'],3,arrow=False)
    path([(2804,920),(2858,920)],C['line'],3.2)
    txt(2650,1336,'P: B × 4 × 4 × 23',20,C['muted'])
    S.append('</g>')
    # Routing controls the sum of candidate predictions, never the latent features.
    S.append('<g id="moe-prediction-fusion">')
    path([(2740,338),(2899,338),(2899,872)],C['pink'],3,True)
    txt(2928,642,'π',29,C['pink'],650)
    circle(2899,920,43,'white',C['pink'],2.7)
    txt(2899,934,'Σ',41,C['pink'],550)
    if lang=='en':
        txt(2899,995,'Prediction',24,C['pink'],600)
        txt(2899,1026,'fusion',24,C['pink'],600)
    else:txt(2899,1003,L['mixture'],24,C['pink'],600)
    path([(2943,920),(2990,920)],C['pink'],3)
    S.append('</g>')
    txt(2455,1378,'Ŷₕ = Σₖ πₖ(x) · Pₕ,ₖ',27,C['ink'],550)
    S.append('</g>')
    # Forecast rows are horizons; columns are commodities, not experts.
    S.append('<g id="final-forecasts">')
    panel(2990,700,280,433,'',C['ink'],'#fafbfd')
    txt(3034,749,L['prediction'],27,C['ink'],650,'start')
    for j,horizon in enumerate([1,5,10,20]):
        yy=810+j*58
        txt(3025,yy+21,str(horizon),26,C['ink'],600)
        for k in range(7):rect(3063+k*25,yy,17,27,C['blue'],C['blue'],2,opacity=.25+.6*((j*3+k*5)%11)/10)
    txt(3130,1073,L['output_shape'],25,C['ink'],600)
    txt(3130,1107,'23 '+L['commodities'],21,C['muted'])
    txt(3114,1198,L['periods'],22,C['muted'])
    S.append('</g>')
    # Minimal visual key instead of a paragraph inside the figure.
    path([(48,1436),(111,1436)],C['line'],3)
    txt(128,1445,L['feature'],24,C['muted'],anchor='start')
    path([(601,1436),(664,1436)],C['pink'],3,True)
    txt(683,1445,L['control'],24,C['muted'],anchor='start')
    txt(1740,1445,L['caption'],24,C['muted'])
    S.append('</svg>')
    svg='\n'.join(S)
    ET.fromstring(svg)
    (OUT/f'c4_parallel_{lang}.svg').write_text(svg)
    return svg


def render(lang):
    import gi
    import cairo
    gi.require_version('Rsvg','2.0')
    from gi.repository import Rsvg
    handle=Rsvg.Handle.new_from_file(str(OUT/f'c4_parallel_{lang}.svg'))
    view=Rsvg.Rectangle();view.x=view.y=0;view.width=W;view.height=H
    for suffix,scale in [('',1.8),('_preview',.6)]:
        surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,int(W*scale),int(H*scale))
        ctx=cairo.Context(surface);ctx.scale(scale,scale);handle.render_document(ctx,view)
        surface.write_to_png(str(OUT/f'c4_parallel_{lang}{suffix}.png'));surface.finish()
    surface=cairo.PDFSurface(str(OUT/f'c4_parallel_{lang}.pdf'),1188,1188*H/W)
    ctx=cairo.Context(surface);ctx.scale(1188/W,1188/W)
    handle.render_document(ctx,view);ctx.show_page();surface.finish()


def main():
    OUT.mkdir(exist_ok=True)
    for lang in LABELS:draw(lang);render(lang)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    names=['models/c4.py','models/backbone.py','models/temporal.py','models/experts.py','models/router.py','models/_native/native.py','models/_native/temporal.py']
    source={str(Path('src/cmgm_c4')/n):sha(ROOT/'src/cmgm_c4'/n) for n in names}
    current=json.loads((ROOT/'configs/project.json').read_text())
    assert current['shape']==dict(nodes=311,stocks=263,bonds=25,targets=23,features=26)
    assert current['horizons']==[1,5,10,20]
    meta=dict(date='2026-11-18',model='moe_phase_m/pred_global_dynamic',layout_version=2,
        previous_layout_preserved='layout_v1/',
        current_training_defaults={k:current['train'][k] for k in ['lr','weight_decay']},
        source_sha256=source,rendering='Original deterministic SVG icons + librsvg/pycairo; editable vector PDF/SVG',
        styling_references=[dict(title='Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks',venue='KDD 2020',figure='Figure 2',url='https://www.xiaojun.ai/papers/kdd-2020-wu.pdf'),
            dict(title='Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting',venue='International Journal of Forecasting 2021',figure='Figure 2',url='https://arxiv.org/pdf/1912.09363')],
        semantic_checks=dict(parallel_space_time=True,macro_drives_micro=True,regimes3_experts4=True,
            live_H_Z_direct_taps=True,A0_global_not_input_conditioned=True,graph_blocks2_each_K2=True,
            no_D0B_gate=True,no_GraphSkip=True,no_horizon_router=True,
            shared_head=True,prediction_level_fusion=True,router_weights_shared_across_horizons=True,
            router_experts_head_and_fusion_inside_one_MoE=True),
        input_shape=['B',20,311,26],candidate_shape=['B',4,4,23],routing_unique_shape=['B',4],
        routing_broadcast_shape=['B',4,4],output_shape=['B',4,23],
        schematic_only=['Heatmap tiles, graph nodes and router bars illustrate structure; they are not trained numerical results'],
        data_or_checkpoint_opened=False,training=False,model_source_modified=False,old_figure_modified=False,
        drawing_script_sha256=sha(Path(__file__)),
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='provenance.json'})
    (OUT/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print('WROTE',OUT)


if __name__=='__main__':main()
