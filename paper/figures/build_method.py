"""Keep the existing illustration and typeset the routing score in matching math."""
from pathlib import Path
import os,re
import matplotlib as mpl
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MPath
mpl.rcParams['mathtext.fontset']='cm'
root=Path(__file__).resolve().parent
svg=root/'method.svg'
text=svg.read_text(encoding='utf-8')
path=TextPath((0,0),r'$s_\alpha$',size=30)
commands=[]
for pts,code in path.iter_segments():
    if code==MPath.CLOSEPOLY: commands.append('Z')
    else: commands.append({MPath.MOVETO:'M',MPath.LINETO:'L',MPath.CURVE3:'Q',MPath.CURVE4:'C'}[code]+' '+' '.join(map(str,pts)))
box=path.get_extents()
element=f'<path aria-label="p_R-p_H" fill="#2B3540" transform="translate({1515-(box.x0+box.x1)/2} 475) scale(1 -1)" d="'+ ' '.join(commands)+'"/>'
text,n=re.subn(r'<path aria-label="p_R-p_H"[^>]*/>',lambda _:element,text)
assert n==1
text=text.replace('(b) Router: five inputs, 4 states','(b) Learned Routing (LR)').replace('>Probabilities</text>','>Normalized score</text>')
svg.write_text(text,encoding='utf-8')
# Windows Cairo dependency; other systems can use their normal Cairo installation.
if os.name=='nt':os.environ['PATH']='D:/miniforge3/envs/autodrama/Library/bin;'+os.environ['PATH']
import cairosvg
cairosvg.svg2pdf(url=str(svg),write_to=str(root/'method.pdf'))
