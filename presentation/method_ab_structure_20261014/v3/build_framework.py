"""Proposal-centred A+B framework with explicit logical protocol roles."""
import render_engine as e


def label(s,x,y,en,zh=None,size=8.3,c='Ink',bold=False,w=None):
    s.text(x,y,en,zh,size,c,bold,w)


def photo_pair(s,x,y):
    s.photo(x+.16,y-.11,'butterfly.png',.60,.60)
    s.rect(x-.02,y+.02,.60,.60,'White','Muted',lw=.4)
    s.photo(x-.02,y+.02,'apple.png',.60,.60)


def factor(s,x,y,name,color,train=None,w=.55,h=.43):
    s.rect(x,y,w,h,color+'Light',color,lw=.65)
    label(s,x+w/2,y+h/2,name,size=9.0)
    if train is True: s.flame(x+w-.04,y-.10,.09)
    if train is False: s.snow(x+w-.04,y-.10,.085)


def basis(s):
    s.rect(.05,.05,17.90,1.82,'Soft','Line',.14,lw=.6)
    label(s,2.95,.33,'Shared CLIP backbone','共享 CLIP 底座',8.8,bold=True,w=5.4)
    # Compact but real components: both encoders frozen, visual LoRA trainable.
    label(s,.80,.84,'[CLASS]','[CLASS]',8.3)
    label(s,.80,1.47,'Image','图像',8.3)
    for y,en,zh in ((.84,'Text enc.','文本编码器'),(1.47,'Image enc.','图像编码器')):
        s.line([(1.40,y),(1.76,y)],arrow=True,lw=.6)
        s.polygon([(1.80,y-.23),(3.45,y-.17),(3.45,y+.17),(1.80,y+.23)],'BlueLight','Blue',.55)
        label(s,2.62,y,en,zh,8.3,w=1.52)
        s.snow(1.77,y-.25,.07)
        s.line([(3.49,y),(4.36,y),(4.36,1.16 if y<1 else 1.25)],lw=.6)
    s.ellipse(4.36,1.20,.39,.39,'White','Muted',.6)
    label(s,4.36,1.20,r'$\cos$',size=8.3)
    s.line([(4.60,1.20),(4.93,1.20)],arrow=True,lw=.6)
    label(s,5.42,1.20,'LA loss','LA 损失',8.3,w=1.15)
    s.rect(3.57,1.48,.66,.27,'OrangeLight','Orange',lw=.55)
    label(s,3.90,1.615,'LoRA',size=8.3)
    s.line([(3.54,1.615),(3.35,1.615)],'Orange',arrow=True,lw=.6)
    s.line([(5.4,1.43),(5.4,1.72),(4.29,1.72)],'Orange',arrow=True,dashed=True,lw=.55)
    s.line([(6.17,.27),(6.17,1.67)],'Line',lw=.5)
    label(s,10.00,.33,'Within-round order','轮内顺序',8.8,bold=True,w=6.5)
    # Routine training is small and neutral; coloured boxes locate the insets.
    nodes=[(6.49,2.15,'Local B + joint','本地 B＋joint','Soft','Muted'),
           (8.99,1.78,'B feedback','B 反馈修正','OrangeLight','Orange'),
           (11.12,2.15,'Local A + joint','本地 A＋joint','Soft','Muted'),
           (13.62,1.78,'A refinement','A 保持修正','BlueLight','Blue')]
    for x,w,en,zh,fill,stroke in nodes:
        s.rect(x,.80,w,.51,fill,stroke,.03,lw=.65)
        label(s,x+w/2,1.055,en,zh,8.3,stroke,False,w-.13)
    for a,b in zip(nodes,nodes[1:]):
        s.line([(a[0]+a[1]+.04,1.055),(b[0]-.06,1.055)],arrow=True,lw=.6)
    s.line([(15.49,1.055),(15.72,1.055),(15.72,1.57),(6.39,1.57),(6.39,1.055),(6.44,1.055)],'Muted',arrow=True,lw=.5)
    s.rect(9.95,1.42,2.10,.30,'Soft','Soft',lw=0)
    label(s,11.0,1.57,'Next round','下一轮',8.3,'Muted')
    s.flame(16.16,.70,.10); label(s,16.97,.70,'Trainable','可训练',8.3)
    s.snow(16.16,1.25,.095); label(s,16.97,1.25,'Frozen','冻结',8.3)


def ordinary(s,dx,f):
    c='Orange' if f=='B' else 'Blue'
    label(s,dx+4.40,3.01,'joint combines local updates','joint：组合客户端本地更新',8.3,'Muted',w=7.6)
    factor(s,dx+.46,3.72,'$'+f+'_t$',c,w=.76,h=.47)
    s.line([(dx+1.27,3.955),(dx+1.68,3.955)],arrow=True,lw=.6)
    s.rect(dx+1.73,3.55,1.70,.79,'White','Line',.03,lw=.6)
    label(s,dx+2.58,3.32,'Clients','客户端',8.3,'Muted')
    label(s,dx+2.58,3.955,'Local '+f+r'\\training',r'本地 '+f+r'\\训练',8.3,w=1.54)
    s.line([(dx+3.48,3.955),(dx+3.89,3.955)],arrow=True,lw=.6)
    s.rect(dx+3.94,3.55,1.52,.79,'White','Line',.03,lw=.6)
    label(s,dx+4.70,3.32,'Server','服务器',8.3,'Muted')
    label(s,dx+4.70,3.955,r'joint\\aggregation',r'joint\\聚合',8.3,w=1.38)
    s.line([(dx+5.51,3.955),(dx+6.10,3.955)],arrow=True,lw=.6)
    factor(s,dx+6.16,3.64,'$'+f+r'_{\rm ord}$',c,w=.94,h=.63)
    label(s,dx+7.84,3.955,'Proposal','提案',8.3,'Muted',w=1.40)


def panel_b(s):
    s.rect(.05,2.07,8.81,9.84,'OrangeBg','Line',.18,lw=.8)
    s.rect(.30,4.81,8.31,2.04,'White','Orange',.08,lw=.7)
    s.rect(.30,7.45,8.31,3.87,'White','Line',.08,lw=.6)
    label(s,4.45,2.42,'B. Feedback on the Shared Proposal','B．聚合后共享提案的直接反馈',9.4,'Orange',True,8.2)
    ordinary(s,0,'B')
    s.line([(6.63,4.32),(6.63,4.54),(.55,4.54),(.55,5.825),(1.10,5.825)],'Orange',arrow=True,lw=.8)
    label(s,4.40,5.08,'SERVER: refine one shared B','服务器：修正同一个共享 B',8.7,'Orange',True,7.8)
    # All clients see exactly this candidate; only delta B is trainable.
    s.rect(1.16,5.46,4.81,.73,'OrangeBg','Line',.04,lw=.6)
    factor(s,1.37,5.64,'$A$','Blue',False)
    factor(s,2.29,5.59,r'$B_{\rm ord}$','Orange',None,w=.85,h=.52)
    label(s,3.55,5.855,'$+$',size=10)
    factor(s,4.09,5.59,r'$\Delta B$','Orange',True,w=.86,h=.52)
    s.line([(6.03,5.855),(7.08,5.855)],'Orange',arrow=True,lw=.85)
    label(s,6.56,5.43,'2 steps','两步',8.3,'Muted',w=1.25)
    s.matrix(7.23,5.56,.64,.58,'Orange')
    label(s,7.55,6.44,r'$B_{\rm new}$',size=9.0)
    label(s,3.03,6.54,'Bounded B update','有界 B 补充更新',8.3,'Muted',w=2.80)
    # Broadcast the shared parameters; no local proposals are re-aggregated here.
    s.line([(1.68,6.24),(1.68,8.10)],'Muted',lw=.7)
    s.rect(.85,6.91,4.10,.36,'OrangeBg','OrangeBg',lw=0)
    label(s,2.90,7.09,'Same shared parameters','相同的共享参数',8.3,'Muted',w=3.95)
    label(s,4.45,7.73,'CLIENTS WITH TAIL DATA','客户端：含尾类的训练数据',8.7,bold=True,w=7.8)
    s.line([(1.55,8.10),(6.75,8.10)],'Muted',lw=.7)
    for x,lab in ((1.55,'1'),(4.15,'k'),(6.75,'K')):
        s.line([(x,8.10),(x,8.42)],'Muted',arrow=True,lw=.7)
        s.monitor(x,8.77,w=1.02)
        label(s,x,9.44,'Client '+lab,'客户端 '+lab,8.3,w=1.8)
        s.data_tiles(x-.38,9.70)
        s.line([(x,9.91),(x,10.09)],'Orange',arrow=True,lw=.65)
        s.ellipse(x,10.34,1.28,.46,'White','Orange',.7)
        label(s,x,10.34,'LA loss','LA 损失',8.3)
        s.line([(x,10.60),(x,10.82)],'Orange',lw=.75)
    s.line([(1.55,10.82),(8.28,10.82),(8.28,6.67),(4.53,6.67),(4.53,6.23)],'Orange',arrow=True,lw=.85)
    label(s,7.36,7.09,'Gradients','梯度',8.3,'Orange',w=1.8)
    label(s,4.14,11.08,'Sum feedback gradients','汇总共享模型上的反馈梯度',8.3,'Orange',w=6.3)
    for j in range(3): s.rect(.53+j*.22,11.53,.12,.22,'OrangeLight','Orange',lw=.4)
    label(s,4.87,11.65,'Equal tail-class loss mass; non-tail unchanged','各尾类损失总权重相等；非尾类不变',8.3,'Muted',w=7.18)


def panel_a(s):
    s.rect(9.04,2.07,8.91,9.84,'GreenBg','Line',.18,lw=.8)
    s.rect(9.29,4.81,8.41,2.04,'White','Blue',.08,lw=.7)
    s.rect(9.29,7.45,8.41,3.87,'White','Line',.08,lw=.6)
    label(s,13.49,2.42,'A. Retention with Continued Learning','A．基于可靠历史的保持与学习',9.4,'Blue',True,8.3)
    ordinary(s,9,'A')
    s.line([(15.63,4.32),(15.63,4.56),(17.46,4.56),(17.46,5.31),(14.10,5.31),(14.10,5.53)],'Blue',arrow=True,lw=.8)
    label(s,13.49,5.08,'SERVER: refine A around its proposal','服务器：在普通提案附近修正 A',8.7,'Blue',True,7.9)
    label(s,10.60,5.50,'History','可靠历史',8.3,'Muted',w=1.8)
    for j in range(5):
        for v in range(2):
            s.rect(9.87+j*.25,5.81+v*.21,.18,.15,'BlueLight','Blue',lw=.4)
    s.line([(11.12,5.98),(11.56,5.98)],'Blue',arrow=True,lw=.6)
    s.rect(11.61,5.75,.66,.46,'BlueLight','Blue',lw=.65)
    label(s,11.94,5.98,'$P_i$',size=9.0)
    s.rect(13.12,5.58,2.01,.76,'BlueBg','Line',.04,lw=.6)
    factor(s,13.37,5.78,'$A$','Blue',True,w=.59)
    s.line([(14.00,5.995),(14.23,5.995)],arrow=True,lw=.6)
    factor(s,14.27,5.78,'$B$','Orange',False,w=.57)
    s.line([(15.20,5.98),(16.27,5.98)],'Blue',arrow=True,lw=.85)
    label(s,15.76,5.54,'3 steps','三步',8.3,'Muted',w=1.22)
    s.matrix(16.42,5.69,.64,.58,'Blue')
    label(s,16.74,6.55,r'$A_{\rm new}$',size=9.0)
    label(s,10.95,6.53,'Capped margin target','截断的余量目标',8.3,'Muted',w=2.85)
    # Model and targets travel together. Margin observations return to history.
    s.line([(12.32,5.98),(12.64,5.98),(12.64,6.70)],'Muted',lw=.6)
    s.line([(13.20,6.38),(13.20,6.70),(12.64,6.70),(12.64,7.05),(9.67,7.05),(9.67,8.80),(10.02,8.80)],'Muted',arrow=True,lw=.7)
    s.rect(10.16,6.93,2.83,.28,'GreenBg','GreenBg',lw=0)
    label(s,11.57,7.065,'Model + targets','模型与目标',8.3,'Muted',w=2.75)
    label(s,13.49,7.73,'CLIENTS: fixed training witnesses','客户端：固定训练见证',8.7,bold=True,w=7.8)
    photo_pair(s,10.09,8.51)
    label(s,10.70,9.49,'All witnesses','全部见证',8.3,w=2.1)
    # A sample may have a deficit at the ordinary proposal; no claim of its cause.
    s.line([(10.91,8.80),(11.46,8.80),(11.46,8.52),(12.14,8.52)],'Blue',arrow=True,lw=.7)
    label(s,13.02,8.09,'History deficit','相对历史的缺口',8.3,'Blue',w=2.65)
    s.line([(12.20,9.02),(13.75,9.02)],'Muted',lw=.5)
    s.rect(12.93,8.83,.29,.19,'BlueLight','Blue',lw=.6)
    s.line([(12.18,8.55),(13.73,8.55)],'Blue',dashed=True,lw=.6)
    label(s,12.30,8.39,'$P_i$',size=8.3,c='Blue')
    label(s,13.07,9.26,r'$m_i$',size=8.3)
    s.line([(13.47,8.59),(13.47,8.81)],'Orange',arrow=True,both=True,lw=.65)
    s.line([(13.79,8.68),(14.42,8.68)],'Blue',arrow=True,lw=.7)
    s.ellipse(15.07,8.68,1.17,.56,'White','Blue',.7)
    label(s,15.07,8.68,'Retain','保持',8.3,'Blue')
    # Classification includes every witness, not just those with valid history.
    s.line([(11.10,8.80),(11.46,8.80),(11.46,10.05),(12.13,10.05)],'Green',arrow=True,lw=.7)
    label(s,12.30,9.92,'$y$',size=8.3)
    label(s,12.23,10.27,r'$c\ne y$',size=8.3)
    s.rect(12.77,9.84,.40,.17,'GreenLight','Green',lw=.5)
    s.rect(12.77,10.19,.81,.17,'Gray','Muted',lw=.5)
    label(s,13.08,10.67,'Classification','分类学习',8.3,w=2.40)
    s.line([(13.79,10.05),(14.42,10.05)],'Green',arrow=True,lw=.7)
    s.ellipse(15.07,10.05,1.17,.56,'White','Green',.7)
    label(s,15.07,10.05,'Learn','学习',8.3,'Green')
    s.line([(15.70,8.68),(17.35,8.68)],'Blue',dashed=True,lw=.75)
    s.line([(15.70,10.05),(17.35,10.05),(17.35,6.78),(14.06,6.78),(14.06,6.38)],'Blue',arrow=True,dashed=True,lw=.85)
    label(s,16.43,7.09,'Joint gradient','联合梯度',8.3,'Blue',w=2.35)
    # Statistics are returned and stored centrally; only committed-model
    # observations may register history. The dotted rail is a later-round path.
    s.line([(10.02,8.95),(9.86,8.95),(9.86,10.40),(10.00,10.40)],'Muted',arrow=True,lw=.55)
    s.line([(10.58,10.72),(10.58,11.03),(9.42,11.03),(9.42,5.98),(9.79,5.98)],'Muted',arrow=True,dashed=True,lw=.55)
    label(s,10.78,10.40,r'Margin\\statistics',r'余量\\统计',8.3,'Muted',w=1.55)
    label(s,13.98,11.04,'Record after model commitment','模型提交后登记历史',8.3,'Muted',w=5.6)
    label(s,13.49,11.65,'Retain necessary margins while learning','保持必要余量，同时继续学习',8.3,'Muted',w=8.0)


def build(lang):
    s=e.Scene(lang)
    basis(s)
    panel_b(s)
    panel_a(s)
    return s


if __name__=='__main__':
    e.build=build
    e.run()
