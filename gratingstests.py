import numpy as np
from numpy import pi,sqrt,sin,cos,ceil,abs
from wavedata import Wave
from gratings import Grating

if __name__ == '__main__':
    def frequencybandwidth(Δλ,λ): # returns df in GHz for dlambda,lambda in nm, or df in Hz for dlambda,lambda in m
        return Δλ*299792458/λ**2 # in GHz
    def simplephasegrating():
        from gratings import grating
        g = Grating(*grating(100,0.5,padx=20001,padcount=1,gapx=0,phasex=0,x0=0))
        f = g.ftgrating(p0=100,dp=6,normalize=0,res=2001)**2
        ff = g.piphasegrating().ftgrating(p0=100,dp=6,normalize=0,res=2001)**2 # piphasegrating == invert bars,gaps of second half
        norm = f.max()
        Wave.plots(f/norm,ff/norm)
    # def Λ2λft(w):
    #     from sellmeier import polingperiod
    #     λs = np.linspace(1300,1700,201)
    #     Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
    #     return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
    def phasegratingfreq():
        from gratings import grating
        from sellmeier import polingperiod
        # print(Grating([0,2,4,6],[1,3,5,7]).spectralcomb(0.1,L=8))
        λ0,L = 1550,50000
        Λ = polingperiod(λ0,sell='mglnridgewg',Type='zzz')
        print('Λ',Λ)
        gg0 = Grating(*grating(Λ,0.5,padx=L)).reverse(L)
        gs = [gg0.phasegrating(n) for n in [1,2,6]]
        gs += [gg0.phasegrating(2).spectralcomb(2/3)]
        fs = [g.ftgrating(p0=Λ,dp=0.08,normalize=0,res=1001)**2 for g in gs]
        fs = [f/fs[0].max() for f in fs]
        # Wave.plots(*fs,seed=3)
        def Λ2λ(w):
            from sellmeier import polingperiod
            λs = np.linspace(1300,1700,201)
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
        def Λ2f(w):
            from sellmeier import polingperiod
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        ns = ['1 phase section','2 phase sections','6 phase sections','"double slit"']
        # Wave.plots(*[Λ2λ(f) for f in fs],seed=3)
        Wave.plots(*[Λ2λ(f).rename(n) for f,n in zip(fs,ns)],seed=3,x='λ (nm)',grid=1,save=f'{L/1000:g}mm MgLN phase grating spectrum vs wavelength')
        Wave.plots(*[Λ2f(f).rename(n) for f,n in zip(fs,ns)],seed=3,x='Δf (GHz)',grid=1,save=f'{L/1000:g}mm MgLN phase grating spectrum vs frequency')
    def phasegratingvsL():
        from gratings import grating
        from sellmeier import polingperiod
        λ0,L = 1550,50000
        Λ = polingperiod(λ0,sell='mglnridgewg',Type='zzz')
        print('Λ',Λ)
        Ls = np.linspace(5,100,100//5)
        gs = [Grating(*grating(Λ,0.5,padx=L*1000)).reverse(L*1000).phasegrating(2) for L in Ls]
        fs = [g.ftgrating(p0=Λ,dp=0.08,normalize=0,res=1001)**2 for g in gs]
        fs = [f/fs[0].max() for f in fs]
        def Λ2f(w):
            from sellmeier import polingperiod
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        fs = [Λ2f(f).rename(f'{L:g}mm') for f,L in zip(fs,Ls)]
        # Wave.plots(*fs,seed=3,x='Δf (GHz)',grid=1,save=f'MgLN phase grating spectrum vs frequency, L')
        def df(w):
            n = len(w)//2
            return abs( w[n:].xmax() - w[:n].xmax() )
        Wave([df(f) for f in fs],Ls).plot(m=1,x='L (mm)',y='Δf (GHz)',grid=1,save='dfvsL')
        Wave([df(f) for f in fs],Ls).plot(m=1,x='L (mm)',y='Δf (GHz)',grid=1,save='dfvsL, log',log=1)
    def phasegrating2(Ls=[25000,50000],λ0=1550):
        from gratings import grating
        from sellmeier import polingperiod
        def Λ2λ(w):
            λs = np.linspace(1300,1700,201)
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
        def Λ2f(w,λ0):
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        Λ = polingperiod(λ0,sell='mglnridgewg',Type='zzz')
        print('Λ',Λ)
        gg0 = Grating(*grating(Λ,0.5,padx=Ls[0])).reverse(Ls[0])
        gg1 = Grating(*grating(Λ,0.5,padx=Ls[1])).reverse(Ls[1])
        gg2 = gg1.phasegrating(2)
        fs = [g.ftgrating(p0=Λ,dp=0.08,normalize=0,res=1001)**2 for g in [gg0,gg1,gg2]]
        fs = [f/fs[0].max() for f in fs]
        # Wave.plots(*fs,seed=3)
        ns = [f'{Ls[0]/10000:g}cm',f'{Ls[1]/10000:g}cm',f'{Ls[1]/10000:g}cm with π phase shift']
        # Wave.plots(*[Λ2λ(f) for f in fs],seed=3)
        Wave.plots(*[Λ2λ(f).rename(n) for f,n in zip(fs,ns)],seed=3,x='λ (nm)',y='relative SHG efficiency',grid=1,save=f'{Ls[0]/1000:g}mm,{Ls[1]/1000:g}mm MgLN phase grating spectrum vs wavelength')
        Wave.plots(*[Λ2f(f,λ0).rename(n) for f,n in zip(fs,ns)],seed=3,x='Δf (GHz)',y='relative SHG efficiency',grid=1,save=f'{Ls[0]/1000:g}mm,{Ls[1]/1000:g}mm MgLN phase grating spectrum vs frequency')
    def phasegrating3(df=50,λ0=1550,L=50000):
        from gratings import grating,alternatinggrating
        from sellmeier import polingperiod
        def Λ2λ(w):
            λs = np.linspace(1300,1700,201)
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
        def Λ2f(w,λ0):
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        Δf = frequencybandwidth(1,λ0); print('Δf',Δf,'GHz')
        Δλ = df/Δf; print('Δλ',Δλ,'nm')
        Λ0 = polingperiod(λ0-Δλ/2,sell='mglnridgewg',Type='zzz')
        Λ1 = polingperiod(λ0+Δλ/2,sell='mglnridgewg',Type='zzz')
        Λ = Λ0/2+Λ1/2
        print('Λ0',Λ0,'Λ1',Λ1)
        gg0 = Grating(*grating(Λ,0.5,padx=L)).reverse(L).phasegrating(2)
        reps = [1,2,50]
        gg1 = Grating(*alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[0]))
        gg2 = Grating(*alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[1]))
        gg3 = Grating(*alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[2]))
        fs = [g.ftgrating(p0=Λ,dp=0.2 if df>200 else 0.05,normalize=0,res=4001)**2 for g in [gg0,gg1,gg2,gg3]]
        fs = [f/fs[0].max() for f in fs]
        ns = [''] + [f'{L/r/2000:g}mm' for r in reps]
        Wave.plots(*[Λ2λ(f).rename(n) for f,n in zip(fs,ns)],seed=3,x='λ (nm)',y='relative SHG efficiency',legendtext='segment length',
            grid=1,save=f'MgLN alternating grating spectrum vs wavelength, df={df}GHz')
        Wave.plots(*[Λ2f(f,λ0).rename(n) for f,n in zip(fs,ns)],seed=3,x='Δf (GHz)',y='relative SHG efficiency',legendtext='segment length',
            grid=1,save=f'MgLN alternating grating spectrum vs frequency, df={df}GHz')
    def ag2test(reps=1):
        from gratings import alternatinggrating
        p0,p1 = 80,120
        g = Grating(*alternatinggrating(p0,p1,0.5,0.5,1,2500,repeats=reps))
        gg = Grating(*alternatinggrating(p0,p1,0.5,0.5,1,2500,repeats=reps))
        g2 = agwithphase(p0,p1,0,0,2500,reps)
        Wave.plots(g.grating2wave(),2+gg.grating2wave(),4+g2.grating2wave(),m=1)
    def agwithphase(p0,p1,f0,f1,L,repeats=1):
        from gratings import grating
        g0 = Grating(*grating(p0,0.5,padx=L,phasex=f0*p0))
        g1 = Grating(*grating(p1,0.5,padx=L,phasex=f1*p1))
        return g0.alternatinggrating2(g1,L,repeats=repeats)
    def phasegrating4(df=25,λ0=1550,L=50000,reps=1):
        from gratings import grating,alternatinggrating
        from sellmeier import polingperiod
        Δf = frequencybandwidth(1,λ0); print('Δf',Δf,'GHz')
        Δλ = df/Δf; print('Δλ',Δλ,'nm')
        Λ0 = polingperiod(λ0-Δλ/2,sell='mglnridgewg',Type='zzz')
        Λ1 = polingperiod(λ0+Δλ/2,sell='mglnridgewg',Type='zzz')
        Λ = Λ0/2+Λ1/2
        print('Λ0',Λ0,'Λ1',Λ1)
        gg0 = Grating(*grating(Λ,0.5,padx=L)).reverse(L).phasegrating(2)
        # reps = [1,1,1] # reps = [50,50,50]
        # from random import random
        # phases = [random(),random(),random()]
        # gg1 = agwithphase(Λ0,Λ1,0,phases[0],L,reps[0])
        # gg2 = agwithphase(Λ0,Λ1,0,phases[1],L,reps[1])
        # gg3 = agwithphase(Λ0,Λ1,0,phases[2],L,reps[2])
        # gg1 = Grating(*alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[0]))
        # gg2 = Grating(*alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[1]))
        # gg3 = Grating(*alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[2]))
        # fs = [g.ftgrating(p0=Λ,dp=0.2 if df>200 else 0.05,normalize=0,res=4001)**2 for g in [gg0,gg1,gg2,gg3]]
        # ns = [''] + [f'{L/r/2000:g}mm' for r in reps]
        def phases(n): return [i/n for i in range(n)]
        ggs = [gg0] + [agwithphase(Λ0,Λ1,0,f,L,reps) for f in phases(10)]
        ns = ['reference'] + [f'{i}' for i,g in enumerate(ggs)]
        fs = [g.ftgrating(p0=Λ,dp=0.2 if df>200 else 0.05,normalize=0,res=4001)**2 for g in ggs]
        fs = [-i/2+f/fs[0].max() for i,f in enumerate(fs)]
        def Λ2λ(w):
            λs = np.linspace(1300,1700,201)
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
        def Λ2f(w,λ0):
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        Wave.plots(*[Λ2λ(f).rename(n) for f,n in zip(fs,ns)],c='k012345678901234567890123456789',seed=3,x='λ (nm)',
            y='relative SHG efficiency', fontsize=8,
            grid=1,save=f'MgLN alternating grating 2, spectrum vs wavelength, df={df}GHz, segments={2*reps}')
        Wave.plots(*[Λ2f(f,λ0).rename(n) for f,n in zip(fs,ns)],c='k012345678901234567890123456789',seed=3,x='Δf (GHz)',
            y='relative SHG efficiency', fontsize=8,
            grid=1,save=f'MgLN alternating grating 2, spectrum vs frequency, df={df}GHz, segments={2*reps}')
    def phasegratingtest0():
        from gratings import grating
        Λ,L = 30,20000
        gg0 = Grating(*grating(Λ,0.5,padx=L)).reverse(L)
        gg1 = gg0.phasegrating(12)
        f0,f1 = [gg.ftgrating(p0=Λ,dp=1,normalize=0,res=2001)**2 for gg in [gg0,gg1]]
        Wave.plots(f0/f0.max(),f1/f0.max())
    def phasegratingtest():
        from gratings import grating
        Λ,L = 30,20000
        gg0 = Grating(*grating(Λ,0.5,padx=L)).reverse(L)
        gg1 = gg0.phasegrating(12)
        f0,f1 = [gg.ftgrating(p0=Λ,dp=1,normalize=0,res=2001)**2 for gg in [gg0,gg1]]
        Wave.plots(f0/f0.max(),f1/f0.max())
    def consecutivegrating(Λ1,Λ2,L1=2500,L2=2500,ΔL=0,apodize=None): # consecutive poling, grating lengths in mm
        from gratings import grating
        print(f" {Λ1:g}Λ1 {Λ2:g}Λ2")
        a = Grating(*grating(Λ1,0.5,padx=L1,padcount=1,gapx=0,phasex=0,x0=-L1/2-ΔL/2,apodize=apodize))
        b = Grating(*grating(Λ2,0.5,padx=L2,padcount=1,gapx=0,phasex=0,x0=-L2/2+ΔL/2,apodize=apodize))
        fa = a.ftgrating(p0=Λ1,dp=3*abs(Λ1-Λ2),normalize=0,res=1001)**2
        fb = b.ftgrating(p0=Λ2,dp=3*abs(Λ1-Λ2),normalize=0,res=1001)**2
        ab = Grating(a.starts+b.starts,a.ends+b.ends,validate=False) # raw concatenation as before, bars interleave/overlap when ΔL<(L1+L2)/2 (deliberately not binaryor)
        ff = ab.ftgrating(p0=0.5*(Λ1+Λ2),dp=6*abs(Λ1-Λ2),normalize=0,res=1001)**2
        # ffb = ab.ftgrating(p0=Λ2,dp=4*abs(Λ1-Λ2),normalize=0,res=1001)**2
        norm = max(fa.max(),fb.max())
        Wave.plots(fa/norm,fb/norm,ff/norm,c='012',l='3300',scale=(2,1),x='Λ (µm)',y='relative QPM response',legendtext=f"{ΔL/1000:g}mm separation",
            save=f"consecutivegrating {L1}L1 {L2}L2 {ΔL}ΔL" + (" apodized" if apodize else ""))
    def phasegratingtests():
        import sellmeier
        phasegratingfreq()
        phasegratingvsL()
        phasegrating2([20000,40000])
        phasegrating2([25000,50000])
        phasegrating2([20000,50000])
        phasegrating3(500)
        phasegrating3(50)
        phasegrating3(25)
        ag2test(3)
        phasegrating4(500)
        phasegrating4(200)
        phasegrating4(50)
        phasegrating4(25)
        phasegrating4(12.5)
        phasegrating4(50,reps=50)
        phasegrating4(25,reps=50)
        phasegrating4(12.5,reps=50)
    def consecutivegratings():
        # consecutivegrating(Λ1=97,Λ2=113,L1=5000,L2=5000,ΔL=0)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=0)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=2500)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=5000)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=7500)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=10000)
        # consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=5000,L2=5000,ΔL=0)
        # consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=5000,apodize='triangle')
    def chirptest():
        from gratings import kchirpgrating
        f0 = Grating(*kchirpgrating(p0=14.45,p1=14.55,dc=0.5,padx=51000,padcount=1)).ftgrating(p0=14.5,dp=0.2,normalize=1,res=1001)**2
        f0.plot(x='Λ (µm)',fewerticks=1)
    def apodizebandwidth(Λ=100,L=10000):
        from gratings import grating
        apodizes = [None,*'trapezoidal,triangle,asingauss23,asintriangle'.split(',')]
        names = 'unapodized,trapezoidal,triangle,asin-gauss,asin-triangle'.split(',')
        gs = [Grating(*grating(Λ,dc=0.5,padx=L,padcount=1,gapx=0,phasex=0,x0=0,apodize=apodize)) for apodize in apodizes]
        fs = [g.ftgrating(p0=Λ,dp=10,normalize=0,amplitude=0,res=2001).rename(s) for s,g in zip(names,gs)]
        fwhms = [f.fwhm() for f in fs]
        names = [s+f" {fwhm:.2f}µm FWHM" for s,fwhm in zip(names,fwhms)]
        fs = [f.rename(s) for s,f in zip(names,fs)]
        Wave.plots(*[f/fs[0].max() for f in fs],c='k0123',l='30000',log=1,xlim='f',ylim=(1e-4,1),grid=1,x='Λ (µm)',y='relative intensity',fontsize=8,abbrev=1,
            save=f"apodized grating bandwidth, L={L/1000:g}mm")
    def apodizedinterleavedgratingtest():
        Λ1,Λ2 = 17.9,6.878
        L = 20000
        apodize = 'tri'
        Δ = 0.215
        dx = 0.1
        assert apodize in ['tri','gauss23','lin','revlin']
        def amptri(x,Δ=0.2): return np.clip( 1-abs(2*(x-0.5)/Δ), 0, 1) # 0 to 1 to 0, Δ = non-zero fraction
        def ampgauss23(x,Δ=0.2,σ=0.23): return np.exp( -0.5 * (x-0.5)**2 / Δ**2  / σ**2 )
        def amplin(x,Δ=0.2): return 0.5 + np.clip((x-0.5)/Δ, -0.5, 0.5) # 0 at start, 1 at end, Δ = linear fraction
        def amprevlin(x,Δ=0.2): return 0.5 + np.clip((0.5-x)/Δ, -0.5, 0.5)
        def f1(x):
            f = {'tri':amptri,'gauss23':ampgauss23,'lin':amprevlin,'revlin':amplin}[apodize]
            return f(x,Δ)
        def f2(x):
            f = {'tri':amptri,'gauss23':ampgauss23,'lin':amplin,'revlin':amprevlin}[apodize]
            return f(x,Δ)
        xx = np.arange(0,L,dx)
        a1 = -np.sin(2*np.pi*xx/Λ1)
        a2 = -np.sin(2*np.pi*xx/Λ2)
        t1 = 1.* 2/pi * Wave(np.cumsum(2/pi * f1(xx/L)),xx,'target')
        t2 = 1.* 2/pi * Wave(np.cumsum(2/pi * f2(xx/L)),xx)
        i,yy,area1,area2 = 0,0*xx,0*xx,0*xx
        while i<len(xx)-1:
            i += 1
            # toolow = (area1[i-1] + abs(a1[i]) < t1[i] - 1)
            # yy[i] = (+1 if (toolow and 0<a1[i]) else -1)
            # area1[i] = area1[i-1] + a1[i]*yy[i]
            pole1 = ((area1[i-1] + abs(a1[i]) < t1[i] - 1) and 0<a1[i])
            pole2 = ((area2[i-1] + abs(a2[i]) < t2[i] - 1) and 0<a2[i])
            pole = (pole1 or pole2)
            pole = pole1 if abs(a1[i])>abs(a2[i]) else pole2
            # pole = pole1 if f1(xx[i]/L) * abs(np.arcsin(a1[i])) > f2(xx[i]/L) * abs(np.arcsin(a2[i])) else pole2
            polingsign = +1 if pole else -1
            yy[i] = 0.5*(polingsign+1)
            area1[i] = area1[i-1] + a1[i]*polingsign
            area2[i] = area2[i-1] + a2[i]*polingsign
        u = Wave(yy,xx,'yy')
        dc = u.smooth(1000,savgol=0).rename('mean dutycycle')
        w1 = Wave(np.cumsum(a1*(2*yy-1))*dx/L*pi/2,xx,'∫A$_1$').sp(l='0',lw=2)
        w2 = Wave(np.cumsum(a2*(2*yy-1))*dx/L*pi/2,xx,'∫A$_2$').sp(l='0',lw=2)
        # wa = Wave(area1*dx/L*pi/2,xx,'area1').sp(l='3',lw=1)
        wt1 = (t1*dx/L*pi/2).sp(c='k',l='2')
        wt2 = (t2*dx/L*pi/2).sp(c='k',l='2')
        # Wave.plot(dc,w1,w2,wt1,wt2,grid=1,seed=1)
        v1 = w1.smooth(10000).diff().rename('A$_1$')*L/dx
        v2 = w2.smooth(10000).diff().rename('A$_2$')*L/dx
        Wave.plot(dc,v1,v2,w1,w2,wt1,wt2,grid=1,seed=1,lw=1,x='x (µm)',y='relative amplitude')
    def jul25atest():
        from Aug25Agratings import barstarts01A,barends01A,barstarts01B,barends01B,barstarts01C,barends01C,barstarts01D,barends01D
        p1,dp1 = 6.878,0.1
        p1,dp1 = 17.9,0.5
        gs = [(s,Grating(b0,b1)) for s,b0,b1 in [('A',barstarts01A,barends01A),('B',barstarts01B,barends01B),('C',barstarts01C,barends01C),('D',barstarts01D,barends01D)]]
        fs = [g.ftgrating(p0=p1,dp=dp1,normalize=1,res=1001).rename(s)**2 for s,g in gs]
        Wave.plot(*fs,x='Λ (µm)',y='relative intensity',xlim=(p1-dp1/2,p1+dp1/2),ylim=(0,1),grid=1)
        # Wave.plot(*[f.fft().abs().normalize().rename(s) for f,s in zip(fs,['A','B','C','D'])],xlim=(-40,40))
        ws = [g.grating2wave().rename(s) for s,g in gs]
        Wave.plot(*ws,l='0123',x='z (mm)',y='relative grating amplitude',grid=1,xlim=(9900,10100),ylim=(-0.1,1.1),scale=(3,1))
    def gratingamplitudetests():
        from gratings import fixedbarapodized,gaussapodizedgrating
        def cumulativeresponse(w,z):
            return w.trapezoid()/z
        g = Grating(*fixedbarapodized(30,1000,σ=0.23))
        g.gratingamplitude(Λ=30).plot(x='z (mm)',y='relative grating amplitude',grid=1,save='relative grating amplitude of fixedbarapodized grating')
        cumulativeresponse(g.gratingamplitude(Λ=30),1000).plot(x='z (mm)',y='cumulative grating amplitude',grid=1,save='cumulative grating amplitude of fixedbarapodized grating')
        Grating(*gaussapodizedgrating(30,1000,σ=0.23)).gratingamplitude(Λ=30).plot(x='z (mm)',y='relative grating amplitude',grid=1,ylim=(-0.1,1.1),save='relative grating amplitude of gaussapodizedgrating')
    def interleavedenhancementfactor():
        from wavedata import wrange,list2str
        from gratings import interleaveddeff,interleavedd2fromd1
        # old way
        def deff(a): # deff for interleaved gratings where -1<a<1 and a=0 for equal magnitude, relative to full strength qpm
            φs = np.linspace(0,pi/2,1001)
            g = Wave( sin(φs) * np.arcsin(np.clip(a + sin(φs), 0, 1)), φs )
            return g.integrate()*2/pi
        xs = wrange(-1,1,0.001)
        a = [deff(x) for x in xs] # Wave(a,xs).plot()
        w0 = Wave(a,a[::-1])
        φs = np.arctan2(a,a[::-1])
        θs = wrange(0,1,0.05) * pi/2
        aa = [Wave(a,φs)(θ) for θ in θs]
        w = Wave(aa,aa[::-1])
        Wave.plots(w0,w.sp(m='o'),x='d₁',y='d₂',aspect=1,grid=1,xlim=(0,1),ylim=(0,1),clip=0)
        # print(list2str(aa)) # 0 0.0780203 0.153918 0.226646 0.295616 0.360619 0.421755 0.47936 0.533936 0.586109 0.63662 0.686245 0.7349 0.782245 0.827742 0.870612 0.909812 0.94405 0.971803 0.991343 1
        # new improved way. same values but defined exactly on 0≤α≤1 instead of discretely approximated on -1≤a≤1.
        for α in [0,0.5,1]:
            d1,d2 = interleaveddeff(α)
            print(f"α={α:.1f}, d1={d1:.3f}, d2={d2:.3f}")
        # α=0.0, d1=0.000, d2=1.000
        # α=0.5, d1=0.637, d2=0.637
        # α=1.0, d1=1.000, d2=0.000
    def apodizedinterleavedtest():
        from gratings import discreteapodizedinterleavedgrating,geometricapodizedinterleavedgrating
        Λ1,Λ2 = 7,17
        g = Grating(*discreteapodizedinterleavedgrating(Λ1,Λ2,1000,σ=0.23,res=0.1,A1=1,A2=1))
        u1 = g.gratingamplitude(Λ1)
        u2 = g.gratingamplitude(Λ2)
        # Wave.plot(u1.smooth(35).rename(1),u2.smooth(35).rename(2),x='z (µm)',y='relative grating amplitude',grid=1,lw=0.5)
        Wave.plot((u1.trapezoid()/1000).rename(1),(u2.trapezoid()/1000).rename(2),x='z (µm)',y='cumulative grating amplitude',grid=1)

        h = Grating(*geometricapodizedinterleavedgrating(Λ1,Λ2,1000,σ=0.23))
        u3 = h.gratingamplitude(Λ1)
        u4 = h.gratingamplitude(Λ2)
        Wave.plot(u3.rename(3),u4.rename(4),x='z (µm)',y='relative grating amplitude',grid=1,lw=0.5)
        Wave.plot((u3.trapezoid()/1000).rename(3),(u4.trapezoid()/1000).rename(4),x='z (µm)',y='cumulative grating amplitude',grid=1)
        Wave.plot((u1.trapezoid()/1000).rename(1),(u2.trapezoid()/1000).rename(2),(u3.trapezoid()/1000).rename(3),(u4.trapezoid()/1000).rename(4),
                  x='z (µm)',y='cumulative grating amplitude',grid=1)
    def exactinterleavedtest(Λ1=7,Λ2=17):
        from gratings import interleavedgrating,geometricinterleavedgrating,geometricapodizedinterleavedgrating
        g0 = Grating(*interleavedgrating(Λ1,Λ2,1,1000))
        g1 = Grating(*geometricinterleavedgrating(Λ1,Λ2,1000))
        g2 = Grating(*geometricapodizedinterleavedgrating(Λ1,Λ2,1000,f1=1))
        Wave.plots(*[g.grating2wave().rename(i)+2*i for i,g in enumerate([g1,g2])],x='z (µm)',y='poling amplitude',scale=(2,1),grid=1,lw=1)
        Wave.plots(*[g.gratingamplitude(Λ1).rename(i) for i,g in enumerate([g0,g1,g2])],x='z (µm)',y='relative Λ1 amplitude',scale=(2,1),grid=1,lw=1)
        Wave.plots(*[g.gratingamplitude(Λ2).rename(i) for i,g in enumerate([g0,g1,g2])],x='z (µm)',y='relative Λ2 amplitude',scale=(2,1),grid=1,lw=1)
    def legacyinterleavedelectrodetests():
        from gratings import Grating,Legacyinterleavedelectrode,interleavedgrating,apodizedinterleavedgrating,shrinkbars,mergetouchingbars,dropsmallbars,breakupgaps
        ## Legacyinterleavedelectrode tests
        v = Legacyinterleavedelectrode(17.9,6.878,padcount=2,overpole=0.6)
        b0,b1 = interleavedgrating(6.878,17.9,2,2500) # note legacy period order
        b0,b1 = shrinkbars(b0,b1,dx=0.6)
        b0,b1 = mergetouchingbars(b0,b1,tolerance=1)
        b0,b1 = dropsmallbars(b0,b1,tolerance=1)
        b0,b1 = breakupgaps(b0,b1,maxgap=0,barsize=1)
        assert v==Grating(b0,b1) and v.length==5000 and v.periods==(17.9,6.878) and v.isvalid() and v.isinbounds()
        assert min(b-a for a,b in zip(v.starts,v.ends))>1 and min(a-b for a,b in zip(v.starts[1:],v.ends[:-1]))>1 # smallestbar guarantees hold
        assert isinstance(v.invertbarsgaps(),Grating) and not isinstance(v.invertbarsgaps(),Legacyinterleavedelectrode)
        assert repr(v)=='Legacyinterleavedelectrode(17.9,6.878,padcount=2,overpole=0.6)'
        va = Legacyinterleavedelectrode(17.9,6.878,padcount=1,apodize='tri')
        assert va==Grating(*dropsmallbars(*mergetouchingbars(*apodizedinterleavedgrating(6.878,17.9,1,2500,apodize='tri'),tolerance=1),tolerance=1))
        assert repr(va)=="Legacyinterleavedelectrode(17.9,6.878,padcount=1,apodize='tri')"

    # phasegratingtests() # ~1hr run time
    if 1:
        simplephasegrating()
        phasegratingtest0()
        phasegratingtest()
        consecutivegratings()
        chirptest()
        apodizebandwidth(106.383)
        apodizedinterleavedgratingtest()
        # jul25atest()
        gratingamplitudetests()
        interleavedenhancementfactor()
        apodizedinterleavedtest()
        exactinterleavedtest()
        legacyinterleavedelectrodetests()