"""Model-first framework: CLIP + alternating LoRA above two mechanism insets."""
import render_engine as e


def encoder(s, x, y, label, zh, fill='BlueLight'):
    s.polygon([(x,y-.59),(x+1.63,y-.39),(x+1.63,y+.39),(x,y+.59)],fill,'Blue',.7)
    s.text(x+.81,y,label,zh,8.6,bold=True,width=1.48)
    s.snow(x-.10,y-.45,.10)


def photos(s,x,y,scale=.69):
    s.rect(x+.20,y-.13,scale,scale,'White','Line',lw=.45)
    s.photo(x+.20,y-.13,'butterfly.png',scale,scale)
    s.rect(x-.04,y+.04,scale,scale,'White','Muted',lw=.45)
    s.photo(x-.04,y+.04,'apple.png',scale,scale)


def training_stage(s,x,factor):
    c='Orange' if factor=='B' else 'Blue'
    s.rect(x,1.10,3.88,3.39,'White','Line',.10,lw=.6)
    s.text(x+1.94,1.40,f'Local {factor} training',f'本地 {factor} 训练',8.9,c,True)
    # The same LoRA linear layer, under two different gradient masks.
    sx=x+.23; sy=2.52
    s.line([(sx,sy),(x+1.21,sy)],arrow=True)
    s.ellipse(x+.43,sy,.05,.05,'Ink','Ink',.25)
    s.line([(x+.43,sy),(x+.43,1.99),(x+.82,1.99)],arrow=True)
    s.rect(x+.85,1.77,.61,.44,'BlueLight','Blue',lw=.65)
    s.text(x+1.155,1.99,'$A$',size=9.5)
    s.line([(x+1.49,1.99),(x+2.12,1.99)],arrow=True)
    s.rect(x+2.15,1.77,.61,.44,'OrangeLight','Orange',lw=.65)
    s.text(x+2.455,1.99,'$B$',size=9.5)
    s.line([(x+2.80,1.99),(x+3.34,1.99),(x+3.34,2.35)],arrow=True)
    s.rect(x+1.24,2.29,1.08,.46,'White','Muted',lw=.65)
    s.text(x+1.78,2.52,'$W^0$',size=9)
    s.snow(x+2.25,2.27,.095)
    s.line([(x+2.35,2.52),(x+3.17,2.52)],arrow=True)
    s.ellipse(x+3.34,2.52,.29,.29,'White','Ink',.65)
    s.text(x+3.34,2.52,'$+$',size=9.5)
    if factor=='B':
        s.snow(x+.80,1.70,.10); s.flame(x+2.80,1.68,.11)
    else:
        s.flame(x+.80,1.68,.11); s.snow(x+2.80,1.70,.10)
    s.line([(x+3.34,2.70),(x+3.34,2.94),(x+1.94,2.94),(x+1.94,3.04)],'Muted',arrow=True)
    s.rect(x+.74,3.07,2.40,.42,'Soft','Muted',.03,lw=.6)
    s.text(x+1.94,3.28,'joint aggregation','joint 聚合',8.3,width=2.23)
    s.line([(x+1.94,3.53),(x+1.94,3.74)],c,arrow=True)
    s.rect(x+.48,3.78,2.92,.51,c+'Light',c,.04,lw=.8)
    en='B: Shared feedback' if factor=='B' else 'A: Retain + learn'
    zh='B：共享反馈修正' if factor=='B' else 'A：保持与学习'
    s.text(x+1.94,4.035,en,zh,8.4,c,True,width=2.74)


def build(lang):
    s=e.Scene(lang)
    # Upper level: actual model architecture plus its alternating training masks.
    s.rect(.04,.04,17.92,5.06,'BlueBg','Line',.25,lw=.8)
    s.text(9,.37,'Alternating Federated LoRA Training','交替联邦 LoRA 训练',10.1,bold=True,width=16)
    s.text(3.65,.88,'Local CLIP model on client k','客户端 k 上的 CLIP 模型',8.8,bold=True,width=6.8)
    s.flame(13.5,.79,.12); s.text(14.28,.81,'Trainable','可训练',8.3)
    s.snow(15.60,.81,.115); s.text(16.35,.81,'Frozen','冻结',8.3)
    # CLIP text path.
    s.rect(.29,1.28,2.03,.71,'White','Muted',.02,lw=.6)
    s.text(1.305,1.635,r'a photo of a\\{[CLASS]}',r'a photo of a\\{[CLASS]}',8.3,width=1.90)
    s.line([(2.36,1.635),(2.74,1.635)],arrow=True)
    encoder(s,2.82,1.635,r'Text\\encoder',r'文本\\编码器')
    s.line([(4.49,1.635),(5.08,1.635)],arrow=True)
    s.rect(5.14,1.31,.25,.65,'GreenLight','Green',lw=.65)
    s.text(5.30,2.22,'Text features','文本特征',8.3,width=2.0)
    # CLIP image path, with an explicit attached LoRA module.
    photos(s,.76,3.13,.75)
    s.text(1.20,4.17,'Local images','本地图片',8.3,width=2.1)
    s.line([(1.83,3.49),(2.74,3.49)],arrow=True)
    encoder(s,2.82,3.49,r'Image\\encoder',r'图像\\编码器')
    s.rect(3.10,4.11,1.05,.43,'OrangeLight','Orange',.02,lw=.7)
    s.text(3.625,4.325,'LoRA',size=8.6,bold=True)
    s.line([(3.625,4.08),(3.625,4.01)],'Orange',arrow=True)
    s.line([(4.49,3.49),(5.08,3.49)],arrow=True)
    s.rect(5.14,3.16,.25,.65,'OrangeLight','Orange',lw=.65)
    s.text(5.30,4.05,'Image features','图像特征',8.3,width=2.0)
    s.line([(5.44,1.635),(6.36,1.635),(6.36,2.38)],arrow=True)
    s.line([(5.44,3.49),(6.36,3.49),(6.36,2.92)],arrow=True)
    s.ellipse(6.36,2.65,.52,.52,'White','Ink',.7)
    s.text(6.36,2.65,r'$\cos$',size=8.5)
    s.line([(6.66,2.65),(7.17,2.65)],arrow=True)
    for j,w in enumerate((.56,.84,.35)):
        s.rect(7.23,2.20+j*.27,w,.16,'Gray' if j!=1 else 'GreenLight','Muted' if j!=1 else 'Green',lw=.4)
    s.text(7.57,1.92,'Scores','分类分数',8.3)
    s.line([(7.59,3.03),(7.59,3.26)],arrow=True)
    s.ellipse(7.57,3.63,1.25,.61,'White','Muted',.7)
    s.text(7.57,3.63,'LA loss','LA 损失',8.4)
    s.line([(7.57,3.99),(7.57,4.32),(4.20,4.32)],'Orange',arrow=True,dashed=True,lw=.8)
    s.text(5.75,4.71,'Update LoRA factors only','仅更新 LoRA 因子',8.3,color='Muted',width=4.2)
    s.line([(8.40,1.13),(8.40,4.73)],'Line',dashed=True,lw=.55)
    training_stage(s,8.86,'B')
    training_stage(s,13.64,'A')
    # Order attaches to stage boundaries, never to encoder activation tensors.
    s.line([(12.31,4.035),(13.12,4.035),(13.12,1.52),(13.60,1.52)],'Ink',arrow=True,lw=.8)
    s.line([(15.58,4.53),(15.58,4.90),(8.61,4.90),(8.61,1.52),(8.81,1.52)],'Muted',arrow=True,lw=.65)
    s.text(11.39,4.68,'Next round','下一轮',8.3,color='Muted')

    # Lower level: two enlarged mechanisms with explicit parameter targets.
    s.rect(.04,5.28,8.81,6.35,'GreenBg','Line',.22,lw=.8)
    s.rect(9.02,5.28,8.94,6.35,'OrangeBg','Line',.22,lw=.8)
    s.text(4.445,5.64,'A. Retention with Continued Learning','A．已有能力保持与继续学习',9.5,'Blue',True,width=8.3)
    s.text(13.49,5.64,'B. Shared Feedback Refinement','B．客户端反馈驱动的共享修正',9.5,'Orange',True,width=8.3)

    # A: history -> a capped margin floor -> retention feedback.
    s.text(1.57,6.14,'Reliable history','可靠历史',8.6,bold=True,width=2.5)
    for j in range(5):
        for v in range(2):
            s.rect(.64+j*.37,6.49+v*.25,.29,.19,'BlueLight','Blue',lw=.5)
    s.text(1.52,7.18,'5 rounds / 2 views','五轮 / 双视图',8.3,width=2.6)
    s.line([(2.57,6.68),(3.01,6.68)],'Blue',arrow=True)
    # Schematic: a historical margin, capped target, current deficit. No success curve.
    s.text(4.00,6.14,'Margin deficit','判别余量缺口',8.6,bold=True,width=2.4)
    s.line([(3.23,7.18),(5.14,7.18)],'Muted',lw=.55)
    s.rect(3.54,6.37,.29,.81,'Gray','Muted',lw=.55)
    s.rect(4.54,6.98,.29,.20,'BlueLight','Blue',lw=.55)
    s.line([(3.25,6.69),(5.13,6.69)],'Blue',dashed=True,lw=.7)
    s.text(3.06,6.70,'$P_i$',size=8.7,color='Blue')
    s.text(3.68,7.44,'$H_i$',size=8.5)
    s.text(4.68,7.44,'$m_i$',size=8.5)
    s.line([(4.95,6.73),(4.95,6.94)],'Orange',arrow=True,both=True,lw=.8)
    s.line([(5.19,6.69),(5.48,6.69)],'Blue',arrow=True)
    s.ellipse(6.05,6.69,1.11,.64,'White','Blue',.8)
    s.text(6.05,6.69,'Retain','保持',8.6,color='Blue')
    s.text(3.86,7.94,'Preserve a small positive margin','保持必要的正判别余量',8.3,color='Muted',width=5.40)

    # A learning branch includes every fixed witness, not only valid history.
    photos(s,.82,8.79,.65)
    s.text(1.47,9.97,'All fixed witnesses','全部固定见证',8.3,width=2.9)
    s.line([(2.01,9.14),(3.05,9.14)],'Green',arrow=True)
    # Class score illustration, labelled by true versus other class.
    s.text(3.35,8.90,'$y$',size=8.8)
    s.text(3.22,9.29,r'$c\ne y$',size=8.3)
    s.rect(3.70,8.82,.54,.18,'GreenLight','Green',lw=.5)
    s.rect(3.70,9.21,1.04,.18,'Gray','Muted',lw=.5)
    s.text(4.07,9.83,'Class predictions','类别预测',8.3,width=2.6)
    s.line([(4.90,9.14),(5.43,9.14)],'Green',arrow=True)
    s.ellipse(6.05,9.14,1.21,.64,'White','Green',.8)
    s.text(6.05,9.14,'Learn','学习',8.6,color='Green')
    # Both gradient signals affect the SAME trainable A; B remains fixed.
    s.rect(6.86,7.44,1.65,1.20,'White','Muted',.07,lw=.65)
    s.rect(7.00,7.80,.58,.48,'BlueLight','Blue',lw=.7)
    s.text(7.29,8.04,'$A$',size=10)
    s.flame(7.26,7.63,.10)
    s.line([(7.61,8.04),(7.79,8.04)],arrow=True,lw=.6)
    s.rect(7.81,7.80,.52,.48,'OrangeLight','Orange',lw=.7)
    s.text(8.07,8.04,'$B$',size=10)
    s.snow(8.09,7.63,.10)
    s.line([(6.66,6.69),(7.29,6.69),(7.29,7.39)],'Blue',arrow=True,dashed=True,lw=.85)
    s.line([(6.69,9.14),(7.29,9.14),(7.29,8.68)],'Green',arrow=True,dashed=True,lw=.85)
    s.line([(7.73,8.69),(8.47,8.69),(8.47,9.67),(7.73,9.67),(7.73,9.94)],'Blue',arrow=True,lw=.75)
    s.text(7.21,9.53,r'3 bounded\\updates',r'三步有限\\修正',8.3,width=1.65)
    s.matrix(7.42,10.01,.62,.57,'Blue')
    s.text(7.73,10.84,'Updated A','更新后的 A',8.4,width=2)
    # Across-round memory is visually secondary to the two gradient branches.
    s.line([(7.73,11.02),(7.73,11.35),(.30,11.35),(.30,6.31),(.63,6.31),(.63,6.43)],
           'Muted',arrow=True,dashed=True,lw=.6)
    s.text(3.82,10.91,'Record committed models for later rounds','记录已提交模型，供后续轮次使用',8.3,color='Muted',width=6.3)

    # B: a common model is broadcast; local training data produces synchronized
    # gradient feedback for one shared delta B, rather than independent B models.
    s.text(12.28,6.13,'One shared model','同一个共享模型',8.7,bold=True,width=4.3)
    s.rect(10.21,6.47,4.12,.78,'White','Muted',.05,lw=.7)
    s.rect(10.48,6.63,.56,.46,'BlueLight','Blue',lw=.65)
    s.text(10.76,6.86,'$A$',size=9.3)
    s.snow(10.48,6.48,.09)
    s.rect(11.38,6.60,.73,.52,'OrangeLight','Orange',lw=.65)
    s.text(11.745,6.86,r'$B_{\rm ord}$',size=8.5)
    s.text(12.54,6.86,'$+$',size=10.5)
    s.rect(13.00,6.60,.78,.52,'OrangeLight','Orange',lw=.7)
    s.text(13.39,6.86,r'$\Delta B$',size=9.0)
    s.flame(13.98,6.63,.11)
    s.line([(14.40,6.86),(15.37,6.86)],'Orange',arrow=True,lw=.9)
    s.text(14.92,6.46,'2 steps','两步',8.3)
    s.matrix(15.64,6.56,.72,.60,'Orange')
    s.text(16.0,7.43,r'$B_{\rm new}$',size=9)
    s.text(16.12,6.17,'Commit','提交',8.4,color='Orange')
    # Downlink tree and a distinct return rail keep the loop easy to trace.
    s.line([(12.28,7.28),(12.28,7.82)],'Muted',lw=.7)
    s.line([(10.55,7.82),(15.91,7.82)],'Muted',lw=.7)
    s.text(10.53,7.51,'To tail-data clients','下发至含尾类客户端',8.3,color='Muted',width=2.92)
    for x,lab,zh in ((10.55,'Client 1','客户端 1'),(13.23,'Client k','客户端 k'),(15.91,'Client K','客户端 K')):
        s.line([(x,7.82),(x,8.28)],'Muted',arrow=True,lw=.7)
        s.monitor(x,8.68,w=1.02)
        s.text(x,9.33,lab,zh,8.3)
        s.data_tiles(x-.38,9.57)
        s.line([(x,9.80),(x,9.89)],'Orange',arrow=True,lw=.65)
        s.ellipse(x,10.14,1.28,.49,'White','Orange',.65)
        s.text(x,10.14,'LA loss','LA 损失',8.3)
        s.line([(x,10.43),(x,10.65)],'Orange',lw=.75)
    s.line([(10.55,10.65),(17.48,10.65),(17.48,7.65),(13.39,7.65),(13.39,7.29)],
           'Orange',arrow=True,lw=.85)
    s.text(13.28,10.89,'Sum client gradients','汇总客户端梯度',8.4,color='Orange',width=4.6)
    # Secondary annotation: weights specify the feedback objective, not the whole B.
    for j in range(3):
        s.rect(9.77+j*.23,11.13,.13,.27,'OrangeLight','Orange',lw=.45)
    s.text(13.39,11.34,'Equal total loss weight per tail class','各尾类的损失总权重相等',8.3,color='Muted',width=6.0)
    return s


if __name__=='__main__':
    e.build=build
    e.run()
