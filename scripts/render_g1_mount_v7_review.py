"""Engineering review figures from the exact native Fusion STL exports."""
from pathlib import Path
import json,sys
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'artifacts/g1_mount_v7'
sys.path.insert(0,str(OUT/'python_packages'))
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle,Polygon

D=json.loads((OUT/'design_inputs.json').read_text(encoding='utf-8'))
BUNDLE=BASE/'artifacts/g1_mount_v7/reference_inputs'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','savefig.facecolor':'#f6f7f9'})
COLORS={'side':'#42836c','crossbar':'#507897'}

def draw(ax,mesh,color,alpha=1):
    if not hasattr(ax,'_g1_scene_parts'):ax._g1_scene_parts=[]
    rgba=list(to_rgba(color));rgba[3]=alpha
    ax._g1_scene_parts.append((mesh.triangles,np.tile(rgba,(len(mesh.faces),1))))

def load(name,assembled=True):
    mesh=trimesh.load_mesh(OUT/'cad'/(name+'_PRINT.stl'))
    if assembled:
        T=np.eye(4);T[:3]=json.loads((OUT/(name+'_fusion.json')).read_text(encoding='utf-8'))['local_to_torso_mm'];mesh.apply_transform(T)
    return mesh

def clean(ax,bounds,elev,azim):
    triangles=np.concatenate([v[0] for v in ax._g1_scene_parts]);colors=np.concatenate([v[1] for v in ax._g1_scene_parts])
    ax.add_collection3d(Poly3DCollection(triangles,facecolors=colors,shade=True,linewidth=0,antialiased=False))
    ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2]);ax.set_box_aspect([b-a for a,b in bounds]);ax.view_init(elev=elev,azim=azim)
    ax.set_axis_off();ax.set_facecolor('#f6f7f9');ax.set_proj_type('ortho')

fig=plt.figure(figsize=(8.5,8.5),facecolor='#f6f7f9')
fig.suptitle('G1 mount · M6 connection',x=.08,y=.96,ha='left',fontsize=22)
fig.text(.08,.91,'Section through the upper-left screw axis · dimensions in mm',color='#526071',fontsize=11)
ax2=fig.add_axes([.14,.20,.80,.62])
mesh=trimesh.load_mesh(OUT/'cad/V7_crossbar_upper_PRINT.stl')
u=D['roots']['upper_left']['center'][1];zrow=D['upper_z'];root_x=D['roots']['upper_left']['center'][0]
section=mesh.section(plane_origin=[u,0,0],plane_normal=[1,0,0])
for path in section.discrete:
    xy=np.column_stack((104-path[:,2],-path[:,1]))
    ax2.fill(xy[:,0],xy[:,1],color=COLORS['crossbar'],alpha=.85)
ax2.add_patch(Rectangle((root_x-6.4,-3),20,6,facecolor='#a9adb3',edgecolor='#454c55',lw=.8))
ax2.add_patch(Rectangle((root_x+12,-6.5),1.6,13,facecolor='#c2c7cd',edgecolor='#454c55',lw=.8))
ax2.add_patch(Rectangle((root_x+13.6,-5),6,10,facecolor='#a9adb3',edgecolor='#454c55',lw=.8))
ax2.plot([root_x+17,109],[0,0],color='#d49b3f',lw=2)
ax2.annotate('Straight Allen-key access',xy=(97,0),xytext=(87,12),arrowprops={'arrowstyle':'->','color':'#9c742f'},fontsize=10,color='#735621')
ax2.annotate('Broad shell seat\nmerged into the crossbar',xy=(root_x+3,-13),xytext=(root_x-12,-30),arrowprops={'arrowstyle':'->','color':'#526071'},fontsize=10,color='#344454')
ax2.annotate('',xy=(root_x,-8),xytext=(root_x+12,-8),arrowprops={'arrowstyle':'|-|','color':'#202b36'})
ax2.text(root_x+6,-11,'12 mm',ha='center',fontsize=9)
ax2.axvline(root_x,color='#7e8790',ls=':',lw=1)
ax2.set(xlim=(root_x-15,110),ylim=(-35,26),aspect='equal')
ax2.set_xlabel('Torso x (mm)');ax2.set_ylabel('Height relative to upper M6 axis (mm)')
ax2.set_title('M6 root section · upper-left',pad=15,fontsize=13)
ax2.spines[['top','right']].set_visible(False);ax2.set_facecolor('#f6f7f9')
fig.text(.08,.06,'M6 screw shown schematically. Actual thread engagement depends on\nthe unmeasured insert recess. Check the fit coupon before full printing.',fontsize=10,color='#526071')
(OUT/'renders').mkdir(exist_ok=True)
fig.savefig(OUT/'renders/m6_root_section.png',dpi=150);plt.close(fig)

fig=plt.figure(figsize=(14,9),facecolor='#f6f7f9')
fig.suptitle('Print orientations · broad faces on the bed',x=.06,y=.97,ha='left',fontsize=21)
names=['V7_side_left','V7_side_right','V7_crossbar_upper','V7_crossbar_lower']
for i,name in enumerate(names):
    ax=fig.add_subplot(2,2,i+1,projection='3d');mesh=load(name,False)
    mesh.apply_translation([-mesh.bounds[:,0].mean(),-mesh.bounds[:,1].mean(),0])
    draw(ax,mesh,COLORS['side' if 'side' in name else 'crossbar'])
    extent=mesh.extents
    clean(ax,[(-150,150),(-150,150),(0,110)],35,-70)
    ax.set_title(name.replace('V7_','').replace('_',' ').title()+f'\n{extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm',fontsize=11,y=.94)
fig.text(.06,.03,'PLA baseline · 0.20 mm layers · 6 walls · 35% gyroid · inspect short hole/counterbore bridges in the slicer.',fontsize=11,color='#526071')
fig.subplots_adjust(top=.9,bottom=.08,hspace=.02,wspace=.02)
fig.savefig(OUT/'renders/print_orientations.png',dpi=150);plt.close(fig)
print('Rendered m6_root_section.png and print_orientations.png')
