"""Build the 16:9 English reliability presentation as a PDF, without PowerPoint."""
import argparse
import io
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT=Path(__file__).resolve().parents[1]
BG,FG,ACCENT='#F7F4EE','#302D2A','#79504B'
W,H=960,540
FONTS=ROOT/'scripts/assets/fonts'
for name,file in [('Body','OpenSans-Regular.ttf'),('Bold','OpenSans-Bold.ttf'),
                  ('Title','LiberationSerif-Regular.ttf'),('TitleBold','LiberationSerif-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/file)))


def read(name):return json.loads((ROOT/name).read_text())

def build(output):
    baseline=read('reports/reliability/baseline/summary.json')
    ablation=read('reports/reliability/ablation/summary.json')
    navigation=read('reports/reliability/navigation/summary.json')
    complete=read('reports/reliability/complete.json')
    c=canvas.Canvas(str(output),pagesize=(W,H),pageCompression=1,invariant=1)
    c.setTitle('CQL in MiniGrid: learning and reliability across ten seeds')
    def text(x,y,value,size=21,font='Body',color=FG):
        c.setFillColor(HexColor(color));c.setFont(font,size)
        for i,line in enumerate(value.split('\n')):
            if pdfmetrics.stringWidth(line,font,size)>W-x-42:raise ValueError('Text exceeds page: '+line)
            c.drawString(x,H-y-i*size*1.4,line)
    def page(title=None):
        if getattr(page,'started',False):c.showPage()
        page.started=True;c.setFillColor(HexColor(BG));c.rect(0,0,W,H,fill=1,stroke=0)
        if title:
            text(48,65,title,36,'Title',ACCENT)
            c.setStrokeColor(HexColor(ACCENT));c.setLineWidth(.8);c.line(48,H-88,912,H-88)
    def bullets(lines,y=145,size=22):
        for line in lines:
            wrapped=textwrap.wrap(line,72 if size==22 else 82)
            text(55,y,'• '+wrapped[0],size)
            for extra in wrapped[1:]:y+=size*1.4;text(77,y,extra,size)
            y+=size*2.5
    def chart(kind):
        plt.rcParams.update({'font.size':12,'text.color':FG,'axes.labelcolor':FG,
                             'xtick.color':FG,'ytick.color':FG,'axes.facecolor':BG,'figure.facecolor':BG})
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        colors={'dqn':'#577B9D','cql':ACCENT,'bc':'#658366'}
        if kind=='baseline':
            for ax,metric,scale,title in zip(axes,['success_rate','return_mean'],[100,1],['Success (%)','Mean return']):
                for i,method in enumerate(['dqn','cql','bc']):
                    rows=[next(r for r in baseline['aggregate'] if r['dataset']==d and r['method']==method) for d in ['strong','mixed']]
                    means=np.array([r[metric]['mean'] for r in rows])*scale;ci=np.array([r[metric]['ci95'] for r in rows])*scale
                    x=np.arange(2)+(i-1)*.24
                    ax.errorbar(x,means,yerr=[means-ci[:,0],ci[:,1]-means],fmt='o',capsize=4,color=colors[method],label=method.upper())
                    for j,d in enumerate(['strong','mixed']):
                        values=[r['metrics'][metric]*scale for r in baseline['runs'] if r['dataset']==d and r['method']==method]
                        ax.scatter(x[j]+np.linspace(-.05,.05,10),values,s=9,color=colors[method],alpha=.55)
                ax.set_xticks([0,1],['Strong','Mixed']);ax.set_title(title);ax.legend(fontsize=9)
                ax.set_ylim((-15,115) if scale==100 else (-1.2,1.2))
        elif kind=='alpha':
            for ax,metric,scale in zip(axes,['success_rate','return_mean'],[100,1]):
                for dataset,color in zip(['strong','mixed_episode','mixed_balanced'],['#577B9D',ACCENT,'#658366']):
                    rows=[next(r for r in ablation['aggregate'] if r['dataset']==dataset and r['method']=='cql' and r['alpha']==a) for a in [0.,.1,1.,5.]]
                    means=np.array([r[metric]['mean'] for r in rows])*scale;ci=np.array([r[metric]['ci95'] for r in rows])*scale
                    ax.errorbar(range(4),means,yerr=[means-ci[:,0],ci[:,1]-means],marker='o',capsize=4,color=color,label=dataset.replace('_',' '))
                ax.set_xticks(range(4),['0','0.1','1','5']);ax.set_xlabel('CQL alpha');ax.set_title('Success (%)' if scale==100 else 'Mean return');ax.legend(fontsize=8)
                ax.set_ylim((-15,115) if scale==100 else (-1.2,1.2))
        else:
            rows=navigation['aggregate']
            for ax,metric,scale in zip(axes,['success_rate','successful_steps_mean'],[100,1]):
                for i,row in enumerate(rows):
                    stat=row[metric];m=stat['mean']*scale;lo,hi=np.array(stat['ci95'])*scale
                    ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=5,color=ACCENT if row['candidate']!='baseline' else '#577B9D')
                    values=[r['metrics'][metric]*scale for r in navigation['runs'] if r['candidate']==row['candidate'] and r['metrics'][metric] is not None]
                    ax.scatter(i+np.linspace(-.08,.08,len(values)),values,color=ACCENT if row['candidate']!='baseline' else '#577B9D',s=16,alpha=.6)
                ax.set_xticks([0,1],['Original CQL','Improved CQL'])
                ax.set_title('Success (%)' if scale==100 else 'Actions on successful episodes')
                if scale==100:ax.set_ylim(-10,110)
        for ax in axes:
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
        stream=io.BytesIO();fig.savefig(stream,format='png',dpi=180,facecolor=BG);plt.close(fig);stream.seek(0)
        c.drawImage(ImageReader(stream),40,85,880,300,preserveAspectRatio=True,anchor='c')
    page();text(48,175,'Offline Reinforcement\nLearning in MiniGrid',49,'Title',ACCENT)
    text(50,320,'Conservative Q-Learning',25)
    text(50,380,'From three training seeds to ten',22)
    text(50,425,'Learn from recorded experience. Test the learned policy.',18)
    page('From experience to action')
    bullets(['Observe the objects, agent direction and elapsed time.',
             'Choose left, right or forward, then receive a reward.',
             'PPO collects transitions; offline methods learn from frozen data.',
             'The final agent chooses actions with its neural network.'])
    page('Problem & environment')
    bullets(['MiniGrid-Dynamic-Obstacles-8x8-v0: reach the green goal.',
             '837 observation values; three actions; a 256-action time limit.',
             'Blocked forward action: −1 and episode termination.',
             'Turning in place is legal; success, collision and timeout differ.'])
    page('Dataset collection')
    bullets(['Strong: 200,035 transitions from final PPO.',
             'Mixed: 200,019 transitions from final PPO, early PPO and random actions.',
             'Balanced: 200,025 transitions with 50% / 25% / 25% source quotas.',
             'Guided navigation: 50,012 transitions. All four frozen files are unchanged.'],size=20)
    # Slide 5 deliberately contains equations only. The recorded action is a_D;
    # b is a distinct summation/maximization variable.
    page()
    equations=[r'$y=r-\eta+\gamma(1-d)\max_{b\in\mathcal{A}}Q_{\bar\theta}(s^{\prime},b)$',
               r'$\mathcal{L}_{TD}=\mathbb{E}_{(s,a_D,r,s^{\prime},d)\sim\mathcal{D}}[\operatorname{Huber}(Q_\theta(s,a_D)-y)]$',
               r'$\mathcal{L}_{C}=\mathbb{E}_{(s,a_D)\sim\mathcal{D}}[\log\sum_{b\in\mathcal{A}}\exp Q_\theta(s,b)-Q_\theta(s,a_D)]$',
               r'$\mathcal{L}_{CQL}=\mathcal{L}_{TD}+\alpha\mathcal{L}_{C}$']
    fig=plt.figure(figsize=(13,6),facecolor=BG)
    for y,e in zip([.82,.60,.38,.16],equations):fig.text(.04,y,e,fontsize=23,color=FG)
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=200,facecolor=BG);plt.close(fig);stream.seek(0)
    c.drawImage(ImageReader(stream),30,50,900,440,preserveAspectRatio=True,anchor='c')
    page('Conservative Q-Learning')
    bullets(['DQN fits recorded actions to temporal-difference targets.',
             'CQL adds a penalty for high values on unsupported actions.',
             'BC directly imitates the recorded actions.',
             'The alpha sweep tests sensitivity; it does not choose alpha from test results.'])
    page('Training & evaluation')
    bullets(['Training seeds 0–9: 60 baseline models and 130 alpha-study conditions.',
             '5,000 updates, batch 256, Adam 0.0003, gamma 0.99; CPU training.',
             '100 common original test environments per model: 6,000 baseline episodes.',
             'Dots show models; intervals show 95% Student t confidence intervals.'],size=20)
    page('Experimental results: ten training seeds');chart('baseline')
    text(50,460,'All seeds retained. Uncertainty is measured across trained models.',18)
    page('Data quality changes the result');chart('alpha')
    text(50,460,'120 CQL conditions + 10 balanced-data BC models; original alpha values.',18)
    page('Improving the navigation policy')
    bullets(['Fine-tune from the original mixed CQL checkpoint of the same seed.',
             'Compare all eight prior step-cost, data and geometry configurations.',
             'Geometry adds 50 features from the existing observation.',
             'Freeze the validation-selected configuration before opening the new test.'],size=20)
    page('Fresh-test navigation results');chart('navigation')
    text(50,460,'New seeds 40000–40199: 2,000 episodes per policy, 4,000 in total.',18)
    page('Conclusions')
    effects=[r for r in baseline['paired_differences'] if r['contrast']=='cql - bc']
    lines=[]
    for r in effects:
        stat=r['statistics']['success_rate'];lo,hi=stat['ci95']
        lines.append(f"{r['dataset'].capitalize()}: CQL − BC = {100*stat['mean']:+.1f} pp; 95% CI [{100*lo:+.1f}, {100*hi:+.1f}].")
    improved=next(r for r in navigation['aggregate'] if r['candidate']!='baseline')
    lines.append(f"Improved navigation: {100*improved['success_rate']['mean']:.1f}% mean success; range {100*improved['success_rate']['minimum']:.1f}–{100*improved['success_rate']['maximum']:.1f}%.")
    lines.append('Results depend on data composition and alpha; ten seeds do not establish universal superiority.')
    bullets(lines,size=20)
    c.save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'docs/RL_Project_Presentation.pdf')
    args=parser.parse_args();build(args.output)
