"""Redraw author-supplied published values, without training or new evaluation.

Input: published_results.csv in this directory.
Output: published_view_count_curves.pdf / .svg, with explicit provenance label.
Dependency: reportlab. PNG is a rasterized preview of the PDF.
"""
from pathlib import Path
import csv
from reportlab.graphics.shapes import Drawing, String, Line, PolyLine, Circle
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "published_results.csv").open()))
assert len(rows) == 42
data = {}
for row in rows:
    data.setdefault(row["dataset"],{}).setdefault(row["method"],{})[int(row["views"])] = {
        "psnr":float(row["psnr_db"]), "ssim":float(row["ssim"]), "lpips":float(row["lpips"])
    }

palette = {
    "RegNeRF":"#8591A4", "SparseNeRF":"#9278B2", "FreeNeRF":"#9278B2",
    "3DGS":"#465D77", "CoR-GS":"#32958D", "DropGaussian":"#C88D22",
    "D2GS":"#3977C5", "SynGS":"#CB3E52",
}
dashes = {
    "RegNeRF":[4,2], "SparseNeRF":[1,2], "FreeNeRF":[1,2],
    "3DGS":[5,2,1,2], "CoR-GS":[4,2], "DropGaussian":[2,2],
    "D2GS":None, "SynGS":None,
}
W,H = 1120,720
d = Drawing(W,H)
ink = colors.HexColor("#1E293B")
muted = colors.HexColor("#66758A")
grid = colors.HexColor("#E5EAF0")

def text(x,y,value,size=10,color=ink,anchor="start",bold=False):
    d.add(String(x,y,value,fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,fillColor=color,textAnchor=anchor))

text(54,684,"SynGS published results: sparse-view comparison",21,bold=True)
text(54,662,"Redrawn from author-supplied Tables 1 and 2. Reported values only; no new experiments.",11,color=muted)

specs = {
    "Mip-NeRF 360":[("psnr","PSNR (dB) - higher is better",(10,32),[10,15,20,25,30]),
                   ("ssim","SSIM - higher is better",(.82,.98),[.84,.88,.92,.96]),
                   ("lpips","LPIPS - lower is better",(0,.24),[0,.06,.12,.18,.24])],
    "LLFF":[("psnr","PSNR (dB) - higher is better",(16,28),[16,19,22,25,28]),
            ("ssim","SSIM - higher is better",(.55,.90),[.60,.70,.80,.90]),
            ("lpips","LPIPS - lower is better",(.06,.34),[.10,.20,.30])],
}

def legend(methods, y):
    entries = []
    for method in methods:
        label = "SynGS (ours)" if method == "SynGS" else method
        width = 28 + stringWidth(label,"Helvetica",10) + 20
        entries.append((method,label,width))
    x = (W-sum(entry[2] for entry in entries))/2
    for method,label,width in entries:
        c = colors.HexColor(palette[method])
        d.add(Line(x,y+3,x+20,y+3,strokeColor=c,strokeWidth=2.4 if method=="SynGS" else 1.3,
            strokeDashArray=dashes[method]))
        d.add(Circle(x+10,y+3,2.4,fillColor=c,strokeColor=c))
        text(x+27,y,label,10,bold=method=="SynGS")
        x += width

for dataset, y, title_y, legend_y in [("Mip-NeRF 360",437,629,375),("LLFF",148,340,77)]:
    text(54,title_y,dataset,14,bold=True)
    for col,(metric,label,(lo,hi),ticks) in enumerate(specs[dataset]):
        x = 64 + 354*col
        width,height = 280,160
        text(x,y+178,label,11,bold=True)
        def px(v): return x+(v-4)/5*width
        def py(v): return y+(v-lo)/(hi-lo)*height
        for tick in ticks:
            ty=py(tick)
            d.add(Line(x,ty,x+width,ty,strokeColor=grid,strokeWidth=.6))
            value = str(int(tick)) if metric=="psnr" else f"{tick:.2f}"
            text(x-10,ty-3,value,9,color=muted,anchor="end")
        for views in (4,6,9):
            tx=px(views)
            d.add(Line(tx,y,tx,y-4,strokeColor=muted,strokeWidth=.7))
            text(tx,y-17,str(views),10,color=muted,anchor="middle")
        d.add(Line(x,y,x+width,y,strokeColor=muted,strokeWidth=.8))
        d.add(Line(x,y,x,y+height,strokeColor=muted,strokeWidth=.8))
        text(x+width/2,y-37,"Input views",10,color=muted,anchor="middle")
        for method,values in data[dataset].items():
            points=[]
            c=colors.HexColor(palette[method])
            for views in (4,6,9):
                val=values[views][metric]
                assert lo <= val <= hi, (dataset,method,metric,val)
                points.extend([px(views),py(val)])
            d.add(PolyLine(points,strokeColor=c,strokeWidth=2.4 if method=="SynGS" else 1.15,
                strokeDashArray=dashes[method],fillColor=None))
            for views in (4,6,9):
                d.add(Circle(px(views),py(values[views][metric]),3.3 if method=="SynGS" else 2.4,
                    strokeColor=c,strokeWidth=.8,fillColor=c))
    legend(list(data[dataset]),legend_y)

text(54,37,"LPIPS = LPIPS* / 100 from the source tables. Three reported view settings; straight segments only.",10,color=muted)
text(54,20,"No per-scene results, error bars, or additional views are inferred. Source archive: P01 / P02, 2026-11-07.",9,color=muted)
pdf = ROOT / "published_view_count_curves.pdf"
svg = ROOT / "published_view_count_curves.svg"
renderPDF.drawToFile(d,str(pdf))
renderSVG.drawToFile(d,str(svg))
print(pdf)
print(svg)
