import numpy as np
from numpy import pi,sqrt,sin,cos,ceil,abs
import sys
from wavedata import Wave

def isvalid(g0,g1=None):
    bs = g0 if g1 is None else grating2bars(g0,g1)
    return all([bi<bj for bi,bj in zip(bs[:-1],bs[1:])])
def grating2bars(starts,ends):
    return [b for p in zip(starts,ends) for b in p]
def bars2grating(bs):
    return bs[0::2],bs[1::2]
def bars2file(file,bars):
    with open(file if file.endswith('.dat') else file+'.dat','w') as f:
        for b in bars:
            f.write(f'{b:.3f}\n')
def grating2xy(starts,ends,delta=0): # convert to plotable format, also used for piecewise linear ft
    # assume starts[n]<ends[n] and ends[n]<starts[n+1]                      # starts,ends = [20,40,60],[25,45,65]
    x = [q+r for p in zip(starts,ends) for q in p for r in [-delta,delta]]  # [20,20,25,25,40,40,45,45,60,60,65,65] or if Δ=1 [19,21,24,26,39,41,44,46,59,61,64,66]
    y = [q for p in zip(starts,ends) for q in [0,1,1,0]]                    # [ 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    return x,y
def gratingfromxys(xys):
    barstarts = [0.5*(x0+x1) for (x0,y0),(x1,y1) in zip(xys[:-1],xys[1:]) if y0<y1]
    barends   = [0.5*(x0+x1) for (x0,y0),(x1,y1) in zip(xys[:-1],xys[1:]) if y0>y1]
    return barstarts,barends
def fixedresgrating2bars(xx,yy): # xx,yy = x position, y grating sign (±1) as np arrays
    y0,y1 = min(yy),max(yy)
    barstarts,barends = [],[]
    if y1==yy[0]: barstarts += [xx[0]]
    for p in range(1,len(xx)):
        if yy[p-1]<y1 and yy[p]==y1:
            barstarts += [xx[p]]
        if y0<yy[p-1] and yy[p]==y0 and not 1==p:
            barends += [xx[p]]
    if y1==yy[-1]: barends += [xx[-1]]
    return barstarts,barends
def signedbars2grating(yy,xx=None,Λ=None): # yy,xx = y grating sign (±1) as np arrays, x position of bar start
    xx = xx if xx is not None else np.arange(len(yy)+1)*Λ/2
    y0,y1 = min(yy),max(yy)
    assert len(xx)==len(yy)+1, 'xx defines positions between bars, yy defines poling of bar'
    assert all([y==y0 or y==y1 for y in yy]), 'expected poled or unpoled, no intermediate values'
    barstarts,barends = [],[]
    ys = [y0]+list(yy)+[y0] # xx[i] is straddled by ys[i],ys[i+1]
    barstarts = [x for i,x in enumerate(xx) if ys[i]<ys[i+1]]
    barends   = [x for i,x in enumerate(xx) if ys[i]>ys[i+1]]
    assert all(b0<b1 for b0,b1 in zip(barstarts,barends))           # check barend follows barstart
    assert all(b0<b1 for b0,b1 in zip(barends[:-1],barstarts[1:]))  # check barstart follows previous barend
    return barstarts,barends
def grating2wave(starts,ends,delta=0):
    x,y = grating2xy(starts,ends,delta)
    return Wave(y,x)
def gratingperiod2wave(starts,ends,delta=0):
    y = np.array(starts[1:]) - np.array(starts[:-1])
    return Wave(y,starts[1:])
def ftgrating(starts,ends,p0=None,dp=None,normalize=True,amplitude=True,res=2001):
    p0 = (np.array(starts[1:])/2-np.array(starts[:-1])/2+np.array(ends[1:])/2-np.array(ends[:-1])/2).mean() if p0 is None else p0
    dp = p0/10 if dp is None else dp
    x,y = grating2xy(starts,ends)
    # w = Wave(y,x)
    w = 2*Wave(y,x)-1
    wp = np.linspace(p0-dp/2,p0+dp/2,res)
    wf = Wave(abs(w.ft(1/wp)),wp)
    # wf = wf if amplitude else wf**2
    wf = wf if amplitude else abs(wf)**2
    return wf/wf[wf.pmax()] if normalize else wf
def grating(period,dc,padx,padcount=1,gapx=0,phasex=0,x0=0,apodize=None,omitpads=()):
    barcount = int(np.ceil(1.*padcount*padx/period))
    barstarts = [x0+phasex+i*period for i in range(barcount)] # startx of each bar
    assert 0<dc and dc<=1
    if gapx:
        def isingap(x): return x%padx + period*dc > padx-gapx # remove if bar end is past gap start
        barstarts = [x for x in barstarts if not isingap(x)]
    def padnum(x): return x//padx
    barstarts = [x for x in barstarts if padnum(x) not in omitpads]
    barends = [x+period*dc for x in barstarts] # endx of each bar
    apodize = {0:None,1:'trapezoidal'}[apodize] if apodize in [0,1] else apodize # keep backwards compatible with Dec20A mask
    if apodize is not None:
        assert 0.5==dc, 'apodization assumes 50% duty cycle'
        assert apodize in 'triangle,trapezoidal,asintriangle,asingauss23'.split(','), apodize
        barstarts,barends = dropsmallbars(*apodizebars(barstarts,barends,apodize=apodize),tolerance=0)
    return barstarts,barends
def apodizebars(starts,ends,afunc=None,apodize=None,σ=0.23): # typical: afunc(0)=2, afunc(0.5)=1, afunc(1)=2 (where 0,1,2 = 0%,50%,100% duty cycle)
    # functions that return 0 to 1 to 0 for x from 0 to 1
    def tri(x): return 1-2*abs(x-0.5) # 0 to 1 to 0
    def trap(x): return np.minimum(1,2*tri(x))
    def gauss(x,σ): return np.exp( -0.5 * (x-0.5)**2 / σ**2 )
    # functions that return duty cycle 1 to 0.5 to 1
    def triangledutycycle(x): return 1-0.5*tri(x)
    def trapezoidaldutycycle(x): return 1-0.5*trap(x)
    def asintriangle(x):
        dc = np.arcsin(tri(x))/np.pi # duty cycle from 0 to 0.5 to 0
        return 1-dc                  # duty cycle from 1 to 0.5 to 1
    def asingauss(x,σ): return 1-np.arcsin(gauss(x,σ))/np.pi
    if afunc is None and apodize is not None:
        afunc = {
            'triangle':     lambda x: 2 * triangledutycycle(x),
            'trapezoidal':  lambda x: 2 * trapezoidaldutycycle(x),
            'asintriangle': lambda x: 2 * asintriangle(x),
            'asingauss23':  lambda x: 2 * asingauss(x,0.23),
            'asingauss':  lambda x: 2 * asingauss(x,σ),
        }[apodize]
    a,b = np.array(starts),np.array(ends)
    c = a/2+b/2                 # c = centers
    w0 = b-a                    # old bar widths
    fp = (c-c[0])/(c[-1]-c[0])  # fp = fractionalposition
    w1 = w0*afunc(fp)           # new bar widths
    return c-w1/2,c+w1/2        # new starts,ends
def mergetouchingbars(barstarts,barends,tolerance=0.0): # if <1.0um won't resolve lithographically, use tolerance=+1.0 to fill in nearly touching bars
    a,b = [barstarts[0]],[]
    for i in range(len(barstarts)-1):
        if barstarts[i+1]-barends[i]>tolerance: # if barstarts[i+1]!=barends[i]:
            a,b = a+[barstarts[i+1]],b+[barends[i]]
    b += [barends[-1]]
    return a,b
def dropsmallbars(barstarts,barends,tolerance=0.0): # if <1.0um won't resolve lithographically, use tolerance=+1.0 to drop bars smaller than this
    a,b = [],[]
    for i in range(len(barstarts)):
        if barends[i]-barstarts[i]>tolerance:
            a,b = a+[barstarts[i]],b+[barends[i]]
    return a,b
def shrinkbars(barstarts,barends,dx):
    return expandbars(barstarts,barends,-dx)
def expandbars(barstarts,barends,op): # positive op → make bars bigger, negative op → make bars smaller
    if callable(op): # op is overpole function, op(barsize) = amount of overpole
        dxs = [op(b1-b0) for b0,b1 in zip(barstarts,barends)] 
        return [a-dx/2 for a,dx in zip(barstarts,dxs)],[b+dx/2 for b,dx in zip(barends,dxs)]
    else: # op = amount of overpole
        return [a-op/2 for a in barstarts],[b+op/2 for b in barends]  # can be negative bar length, so call dropsmallbars()
def invertbarsgaps(starts,ends):
    return ends[:-1],starts[1:]
def breakupbars(barstarts,barends,maxbar,gapsize):
    if not maxbar:
        return barstarts,barends
    assert gapsize<=maxbar
    xx0,xx1 = [],[] # xx0,xx1 will be the new barstarts,barends
    for x0,x1 in zip(barstarts,barends):
        if maxbar < x1-x0:
            n = 1 + int((x1-x0+gapsize)/(maxbar+gapsize))
            gap = (x1-x0+gapsize)/n - gapsize
            xx0 += [x0+i*(gap+gapsize) for i in range(n)]
            xx1 += [x0+i*(gap+gapsize)+gap for i in range(n)]
            assert np.allclose(x1,x0+(n-1)*(gap+gapsize)+gap)
        else:
            xx0,xx1 = xx0+[x0],xx1+[x1]
    return xx0,xx1
def breakupgaps(barstarts,barends,maxgap,barsize):
    gapstarts,gapends = barends[:-1],barstarts[1:]
    xx0,xx1 = breakupbars(gapstarts,gapends,maxbar=maxgap,gapsize=barsize)
    return barstarts[:1]+xx1,xx0+barends[-1:]
def apodizebreakup(Λ,σ,dc,gx,defaulton,minbar):
    g = apodizebars(*grating(Λ,dc=dc,padx=gx),σ=σ,apodize='asingauss')
    gg = g if defaulton else invertbarsgaps(*g)
    return breakupbars(*gg,maxbar=2*minbar,gapsize=minbar)
def kchirpgrating(p0,p1,dc,padx,padcount=1,gapx=0,phase=0): # https://www.gaussianwaves.com/2014/07/chirp-signal-frequency-sweeping-fft-and-power-spectral-density/
    assert 0==phase, 'phase not implemented'
    k0,k1 = 2*pi/p0,2*pi/p1
    barcount = int(padcount*padx*(k1+k0)/4/pi)
    L = 4*pi*barcount/(k1+k0)
    def k(x): return 0.5*(k1-k0)*x/L + k0 # note that ϕ(x) = ∫xk(x)dx so the instantaneous k at x=L is ϕ'(L) = k₁ not ½k₁ as naively expected
    def m(x): return x*k(x)/2/pi
    def xx(m): return ( sqrt(m*4*pi*(k1-k0)/L + k0*k0) - k0 )*L/(k1-k0)
    ms = np.arange(barcount+1)  # period number
    xs = xx(ms)                 # start/end location of each period
    periods = np.diff(xs)       # length of each period
    barstarts = xs[:-1]
    if gapx:
        def isingap(x,period): return x%padx + period*dc > padx-gapx # remove if bar end is past gap start
        barstarts,periods = zip(*[(x,p) for x,p in zip(barstarts,periods) if not isingap(x,p)])
    barends = [x+dc*p for x,p in zip(barstarts,periods)]
    return barstarts,barends
def entwinedgrating(period1,period2,padcount,gx):
    return interleavedgrating(period1,period2,padcount,gx)
def interleavedgrating(period1,period2,padcount,gx,res=0.1):
    # res,minbarsize = 0.1,max(0.6,overpole)
    xx = np.arange(0,1.*padcount*gx,res)
    yy = np.sign( np.sin(2*np.pi*xx/period1) + np.sin(2*np.pi*xx/period2) )
    barstarts,barends = fixedresgrating2bars(xx,yy)
    # if minbarsize:
    #     bars = [(x,y) for x,y in zip(barstarts,barends) if y-x>minbarsize]
    #     barstarts,barends = zip(*bars)
    # return barstarts,[b-overpole for b in barends]
    return barstarts,barends
def apodizedinterleavedgrating(period1,period2,padcount,gx,apodize='triangle',res=0.1):
    def splitnumbers(s: str):
        # e.g. 'abc2.5xxx5e8$' → ['abc', 2.5, 'xxx', 5., 'e', 8., '$']
        import re
        pattern = r'\d+(?:\.\d+)?'                      # integer or decimal
        tokens = re.split(f'({pattern})', s)            # keep the numbers in the list
        return [float(t) if re.fullmatch(pattern, t) else t for t in tokens if t]
    L,Δ = padcount*gx,1
    if apodize[-1] in '0123456789':
        apodize,z = splitnumbers(apodize) # z = length in mm over which it happens, full length by default
        Δ = 1000*z/L
        assert 0<Δ<=1, Δ
    assert apodize in ['tri','gauss','lin','revlin']
    def amptri(x,Δ=0.2): return np.clip( 1-abs(2*(x-0.5)/Δ), 0, 1) # 0 to 1 to 0, Δ = non-zero fraction
    def ampgauss23(x,Δ=0.2,σ=0.23): return np.exp( -0.5 * (x-0.5)**2 / Δ**2  / σ**2 ) # 0 to 1 to 0
    def amplin(x,Δ=0.2): return 0.5 + np.clip((x-0.5)/Δ, -0.5, 0.5) # 0 at start, 1 at end, Δ = linear fraction
    def amprevlin(x,Δ=0.2): return 0.5 + np.clip((0.5-x)/Δ, -0.5, 0.5)
    def f1(x):
        f = {'tri':amptri,'gauss':ampgauss23,'lin':amprevlin,'revlin':amplin}[apodize]
        return f(x,Δ)
    def f2(x):
        f = {'tri':amptri,'gauss':ampgauss23,'lin':amplin,'revlin':amprevlin}[apodize]
        return f(x,Δ)
    xx = np.arange(0,L,res)
    a1 = -np.sin(2*np.pi*xx/period1)
    a2 = -np.sin(2*np.pi*xx/period2)
    t1 = .6* 2/pi * Wave(np.cumsum(2/pi * f1(xx/L)),xx,'target')
    t2 = .6* 2/pi * Wave(np.cumsum(2/pi * f2(xx/L)),xx)
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
    barstarts,barends = fixedresgrating2bars(xx,yy)
    # u = Wave(yy,xx,'yy')
    # dc = u.smooth(1000,savgol=0).rename('mean dutycycle')
    # w1 = Wave(np.cumsum(a1*(2*yy-1))*res/L*pi/2,xx,'∫A$_1$').sp(l='0',lw=2)
    # w2 = Wave(np.cumsum(a2*(2*yy-1))*res/L*pi/2,xx,'∫A$_2$').sp(l='0',lw=2)
    # # wa = Wave(area1*res/L*pi/2,xx,'area1').sp(l='3',lw=1)
    # wt1 = (t1*res/L*pi/2).sp(c='k',l='2')
    # wt2 = (t2*res/L*pi/2).sp(c='k',l='2')
    # # Wave.plot(dc,w1,w2,wt1,wt2,grid=1,seed=1)
    # v1 = w1.smooth(10000).diff().rename('A$_1$')*L/res
    # v2 = w2.smooth(10000).diff().rename('A$_2$')*L/res
    # Wave.plot(dc,v1,v2,w1,w2,wt1,wt2,grid=1,seed=1,lw=1,x='x (µm)',y='relative amplitude')
    # def amplitude(w):
    #     from wavedata import wrange
    #     ux = wrange(0,L,0.1)
    #     wu = Wave([w(x,extrapolate='constant') for x in ux],ux)
    #     return [( pi/2 * (1-2*wu) * np.sin(2*pi*ux/Λ) ).smooth(10000) for Λ in (period1,period2)]
    # w = grating2wave(barstarts,barends)
    # Wave.plot(*amplitude(w),seed=1,lw=1)
    # exit()
    return barstarts,barends
def apodizedinterleavedgrating2(period1,period2,gx,σ=0.23,res=0.1,strength1=0.6,strength2=None):
    strength2 = strength1 if strength2 is None else strength2
    def gaussamplitude(x): return np.exp( -0.5 * (x-0.5)**2  / σ**2 ) # 0 to 1 to 0
    xx = np.arange(0,gx,res)
    a1 = -np.sin(2*pi*xx/period1)
    a2 = -np.sin(2*pi*xx/period2)
    t1 = strength1 * 2/pi * Wave(np.cumsum(2/pi * gaussamplitude(xx/gx)),xx,'Λ1 target')
    t2 = strength2 * 2/pi * Wave(np.cumsum(2/pi * gaussamplitude(xx/gx)),xx,'Λ2 target')
    i,yy,area1,area2 = 0,0*xx,0*xx,0*xx
    while i<len(xx)-1:
        i += 1
        pole1 = ((area1[i-1] + abs(a1[i]) < t1[i] - 1) and 0<a1[i])
        pole2 = ((area2[i-1] + abs(a2[i]) < t2[i] - 1) and 0<a2[i])
        pole = pole1 if abs(a1[i])>abs(a2[i]) else pole2
        polingsign = +1 if pole else -1
        yy[i] = 0.5*(polingsign+1)
        area1[i] = area1[i-1] + a1[i]*polingsign
        area2[i] = area2[i-1] + a2[i]*polingsign
    barstarts,barends = fixedresgrating2bars(xx,yy)
    return barstarts,barends
def interleaveddeff(α):
    from scipy.special import ellipe, hyp2f1
    if not 0<=α<=1: raise ValueError("0≤α≤1 must hold")
    t = (1-α)/α if α>=0.5 else α/(1-α)
    d1,d2 = 2/pi * ellipe(t**2), t/2 * hyp2f1(0.5, 0.5, 2, t**2)
    return (d1,d2) if α>=0.5 else (d2,d1)
def interleavedd2fromd1(d1):
    from scipy.optimize import brentq
    if not 0<=d1<=1: raise ValueError("0≤d1≤1 must hold")
    α = brentq(lambda a: interleaveddeff(a)[0] - d1, 0.0, 1.0)
    return interleaveddeff(α)[1]
def discreteapodizedinterleavedgrating(period1,period2,gx,σ=0.23,res=0.1,A1=None,A2=None):
    A1 = 2/pi if A1 is None else A1
    A2 = interleavedd2fromd1(A1) if A2 is None else A2
    def gaussamplitude(x): return np.exp( -0.5 * (x-0.5)**2  / σ**2 ) # 0 to 1 to 0
    from wavedata import wrange
    xx = wrange(0,gx/2,res)
    a1 = np.cos(2*pi*(xx-gx/2)/period1)
    a2 = np.cos(2*pi*(xx-gx/2)/period2)
    t1 = A1 * Wave(np.cumsum(2/pi * gaussamplitude(xx/gx)),xx,'Λ1 target')
    t2 = A2 * Wave(np.cumsum(2/pi * gaussamplitude(xx/gx)),xx,'Λ2 target')
    i,yy,area1,area2 = 0,0*xx,0*xx,0*xx
    while i<len(xx)-1:
        i += 1
        pole1 = ((area1[i-1] + abs(a1[i]) < t1[i] - 1) and 0<a1[i])
        pole2 = ((area2[i-1] + abs(a2[i]) < t2[i] - 1) and 0<a2[i])
        # pole = pole1 if abs(a1[i])>abs(a2[i]) else pole2
        pole = pole1 if (t1[i]-area1[i-1])*abs(a1[i]) > (t2[i]-area2[i-1])*abs(a2[i]) else pole2
        polingsign = +1 if pole else -1
        yy[i] = 0.5*(polingsign+1)
        area1[i] = area1[i-1] + a1[i]*polingsign
        area2[i] = area2[i-1] + a2[i]*polingsign
    xx,yy = np.concatenate([xx[:-1], gx-xx[::-1]]), np.concatenate([yy[:-1], yy[::-1]])
    return fixedresgrating2bars(xx,yy)
def preferencegrating(Λ1, Λ2, gx, x1=None, x2=None):
    x1,x2 = gx/2 if x1 is None else x1, gx/2 if x2 is None else x2
    # returns exact intervals where Λ1 dominates. e.g. g1=1 if Λ1 dominates, g1=0 if Λ2 dominates
    # dominates means that |cos(2π(x-x1)/Λ1)| > |cos(2π(x-x2)/Λ2)|
    # g1*sign(cos[2π(x-x1)/Λ1]) + (1-g1)*sign(cos[2π(x-x2)/Λ2]) makes an interleaved grating
    a = lambda x: 2 * np.pi * (x - x1) / Λ1
    b = lambda x: 2 * np.pi * (x - x2) / Λ2
    # toggles where |cos a| = |cos b|  ->  sin(a+b) sin(a-b) = 0
    Λsum = Λ1*Λ2 / (Λ1 + Λ2)      # sin(a+b):  a+b = 2π(x/Λsum - θp)
    Λdif = Λ1*Λ2 / (Λ2 - Λ1)      # sin(a-b):  a-b = 2π(x/Λdif - θm)   (signed)
    θp = x1/Λ1 + x2/Λ2
    θm = x1/Λ1 - x2/Λ2
    def lattice(Λ, Φ):            # zeros at  x = (k/2 + Φ) * Λ  within (0, gx)
        k0, k1 = -2*Φ, 2*(gx/Λ - Φ)                       # k at x=0 and x=gx
        k = np.arange(np.floor(min(k0, k1)) - 1, np.ceil(max(k0, k1)) + 2)
        return (k/2 + Φ) * Λ
    edges = np.concatenate([lattice(Λsum, θp), lattice(Λdif, θm)])
    edges = np.sort(edges[(edges > 0) & (edges < gx)])    # = the selector toggle points
    nodes = np.concatenate(([0.0], edges, [gx]))
    mids  = 0.5 * (nodes[:-1] + nodes[1:])
    g1 = np.abs(np.cos(a(mids))) > np.abs(np.cos(b(mids)))   # Λ1 dominates
    return list(nodes[:-1][g1]), list(nodes[1:][g1])
def binaryand(g0, g1): # multiply two binary gratings (assuming bars are 1 and gaps are 0)
    (a0, b0), (a1, b1) = g0, g1
    i,j,starts,ends = 0,0,[],[]
    while i<len(a0) and j<len(a1):
        left  = max(a0[i],a1[j])
        right = min(b0[i],b1[j])
        if left<right:  # nonzero-width intersection
            starts += [left]
            ends += [right]
        i,j = (i+1,j) if b0[i]<b1[j] else (i,j+1) if b1[j]<b0[i] else (i+1,j+1)
    return mergetouchingbars(starts,ends,1e-3)
def binaryor(g0, g1): # add two binary gratings (assuming bars are 1 and gaps are 0 and 1+1=1)
    (a0, b0), (a1, b1) = g0, g1
    i, j, starts, ends = 0, 0, [], []
    while i<len(a0) or j<len(a1):
        if j == len(a1) or (i<len(a0) and a0[i]<=a1[j]):
            left,right = a0[i],b0[i]
            i += 1
        else:
            left,right = a1[j],b1[j]
            j += 1
        if not starts or left>ends[-1]:
            starts += [left]
            ends += [right]
        else:
            ends[-1] = max(ends[-1],right)
    return starts,ends
def geometricapodizedinterleavedgrating(Λ1,Λ2,gx,f1=None,f2=None,σ=None,x1=None,x2=None): # exact bars for apodized sign( cos(2pi x/Λ1) + cos(2pi x/Λ2) )
    # f1,f2 = target ACHIEVED amplitude per channel: a function of x/gx, or a constant multiplying the
    # gaussian(σ) envelope (envelope peak = f, 0 ≤ f ≤ 2/π), defaulting to the max constant 2/π
    def transferfunc(c): # achieved amplitude per channel vs requested amplitude, https://claude.ai/share/8c492cb1-b225-4345-b389-0e86126a6bb1
        c = np.clip(np.asarray(c,dtype=float),0,1)
        return c - 2/pi*( c*np.arcsin(c) + np.sqrt(1-c**2) - 1 )
    def transferfuncinv(d):
        xs = np.linspace(0,1,101)
        return np.interp(d, transferfunc(xs), xs)
    f1 = 2/pi if f1 is None else f1
    f2 = f1 if f2 is None else f2
    assert σ is not None or (callable(f1) and callable(f2)), 'σ required for constant f1,f2'
    def target(f):
        return f if callable(f) else lambda x,f=f: f * np.exp( -0.5 * (x-0.5)**2 / σ**2 )
    r1,r2 = lambda x: transferfuncinv(target(f1)(x)), lambda x: transferfuncinv(target(f2)(x))
    g1,g2 = apodizedgrating(Λ1,gx,f=r1,overfill=1), apodizedgrating(Λ2,gx,f=r2,overfill=1)
    h1,h2 = preferencegrating(Λ1,Λ2,gx,x1=x1,x2=x2),preferencegrating(Λ2,Λ1,gx,x1=x2,x2=x1)
    return binaryor( binaryand(g1,h1), binaryand(g2,h2) )
def geometricinterleavedgrating(Λ1, Λ2, gx, φ1=None, φ2=None): # exact bars for sign( cos(2pi x/Λ1+φ1) + cos(2pi x/Λ2+φ2) )
    φ1,φ2 = -gx*pi/Λ1 if φ1 is None else φ1, -gx*pi/Λ2 if φ2 is None else φ2 # default to bars symmetric about the center gx/2
    # cos a + cos b = 2 cos((a+b)/2) cos((a-b)/2), factor cos(ω x + φ) = 0 at ω x + φ = (k+½)π
    factors = [(np.pi*(1/Λ1 + 1/Λ2), (φ1 + φ2)/2),   # carrier
                (np.pi*(1/Λ1 - 1/Λ2), (φ1 - φ2)/2)]   # envelope (ω may be < 0)
    zeros = []
    for ω, φ in factors:
        k0, k1 = φ/np.pi - 0.5, (ω*gx + φ)/np.pi - 0.5      # k at x=0 and x=gx
        k = np.arange(np.floor(min(k0, k1)) - 1, np.ceil(max(k0, k1)) + 2)
        zeros.append(((k + 0.5)*np.pi - φ) / ω)
    zeros = np.sort(np.concatenate(zeros))
    zeros = zeros[(zeros > 0) & (zeros < gx)]
    nodes = np.concatenate(([0.0], zeros, [gx]))            # interval boundaries
    mids  = 0.5 * (nodes[:-1] + nodes[1:])
    pos   = np.cos(2*np.pi*mids/Λ1 + φ1) + np.cos(2*np.pi*mids/Λ2 + φ2) > 0
    barstarts, barends = list(nodes[:-1][pos]), list(nodes[1:][pos])
    return barstarts, barends
def halfperiodgapgrating(period,n,padcount,gx): # n = number of periods between half-period gaps
    m = int(padcount*gx/(period*0.5*(2*n+1)))
    barstarts = [0.5*(2*i+1) + j*0.5*(2*n+1) for j in range(m) for i in range(n)]
    barstarts = period*np.array(barstarts)
    barends = barstarts + 0.5*period
    return barstarts,barends
def phaseflipgrating(period,n,padcount,gx): # n = number of periods between half-period gaps
    m = 2*int(padcount*gx/(2*n*period))
    def sectionstarts(n,j):
        return [2*i for i in range(1,n)] if j%2 else [2*i+1 for i in range(n)]
    barstarts = [(x+j*2*n)*0.5*period for j in range(m) for x in sectionstarts(n,j)]
    barends =   [(x+j*2*n)*0.5*period for j in range(m) for x in sectionstarts(n,j+1)]
    return barstarts,barends
def alternatinggrating2(g0,g1,L,repeats=1): # L in µm
    def keepit(bar,j):
        x0,x1 = bar
        if 0==j:
            return x0%(L/repeats)<0.5*(L/repeats) and x1%(L/repeats)<0.5*(L/repeats) and 0<=x0 and x1<=L
        else:
            return x0%(L/repeats)>0.5*(L/repeats) and x1%(L/repeats)>0.5*(L/repeats) and 0<=x0 and x1<=L
    b0 = [b for b in zip(*g0) if keepit(b,0)]
    b1 = [b for b in zip(*g1) if keepit(b,1)]
    bars = sorted(b0+b1)
    starts,ends = zip(*bars)
    return starts,ends
def alternatinggrating(p0,p1,dc0,dc1,padcount,gx,repeats=1):
    def bars(k,ends=False): # k = section number
        p,dc = (p0,dc0) if 0==k%2 else (p1,dc1) # p0 for even sections, p1 for odd
        x0,x1 = k*padcount*gx/2./repeats,(k+1)*padcount*gx/2./repeats # startx,endx of section
        n0,n1 = np.ceil(x0/p),np.floor(x1/p)-1 # number of first,last bar (assuming bar 0 is at x=0)
        xs = p*np.arange(n0,n1+1,1) + ends*p*dc # there will always be a small unpoled part at the start and/or end of each section: ___■■__■■__■■__■■____
        # assert xs[0]<x0+p+ends*p*dc, f'{xs[0]} {x0} {p} {x0/p} {ceil(x0/p)}'
        # assert xs[-1]>=x1-2*p, f'{xs[-1]} {x1} {p} {x1/p} {floor(x1/p)} {k} {ends}'
        assert all([x0<=x<=x1 for x in xs]), f'{xs[-1]} {x1} {p} {x1/p} {floor(x1/p)} {k} {ends}'
        # for x in xs: assert x0<=x<=x1, f'{x0},{x},{x1} n0*p:{n0*p} n0:{n0} n1:{n1} section number:{k} ends:{ends}'
        return xs
    b0s = [b for n in range(2*repeats) for b in bars(n,ends=False)]
    b1s = [b for n in range(2*repeats) for b in bars(n,ends=True )]
    return b0s,b1s
def piphasegrating(g0,g1): # g0,g1 = starts,ends
    n = len(g1)//2
    a0,a1 = g0[:n],g1[:n]
    b0,b1 = g0[n:],g1[n:]
    b0,b1 = invertbarsgaps(b0,b1)
    return a0+b0,a1+b1
def reverse(L,g0,g1):
    return [L-g for g in g1[::-1]],[L-g for g in g0[::-1]]
def phasegrating(g0,g1,n,x0=0,x1=None):
        ### recommended method for one less missing bar at end
        # gg = grating(100,0.5,300,padcount=1,gapx=0,phasex=0,x0=0)
        # hh = phasegrating(*reverse(300,*gg),2)
    x1 = x1 if x1 is not None else g1[-1]
    xs = [x0+(x1-x0)*i/n for i in range(1,n)]
    bs = sorted(grating2bars(g0,g1) + xs)
    bs = bs[:len(bs)//2*2]
    gg = bars2grating(bs)
    if isvalid(bs):
        return gg
    return mergetouchingbars(*dropsmallbars(*gg,tolerance=0),tolerance=0)
def flipgratingbars(signs,period,tolerance): # note signed segments are length period/2
    def lacc(*args,**kwargs):
        from itertools import accumulate
        return list(accumulate(*args,**kwargs))
    zs = lacc([period/2. for _ in signs],initial=0)
    signs = [+1] + signs + [+1]
    sign0s,sign1s = signs[:-1],signs[1:] # signs before,after position z
    assert len(zs)==len(signs)-1==len(sign0s)==len(sign1s)
    barstarts = [z for z,sign0,sign1 in zip(zs,sign0s,sign1s) if sign0==+1 and sign1==-1]
    barends = [z for z,sign0,sign1 in zip(zs,sign0s,sign1s) if sign0==-1 and sign1==+1]
    assert barstarts==[z for z,sign0,sign1 in zip(zs,sign0s,sign1s) if sign0>sign1]
    return mergetouchingbars(barstarts,barends,tolerance=tolerance)
def spectralcomb(g0,g1,offfraction,L=None): # removes all bars in middle: ■■____■■, L in µm
    L = L if L is not None else g1[-1]
    x0,x1 = (1-offfraction)*L/2,(1+offfraction)*L/2
    ps = [(a,b) for a,b in zip(g0,g1) if a<x0 or x1<b]
    ps = [(a,min(b,x0) if a<x0 else b) for a,b in ps]
    ps = [(max(a,x1) if x1<b else a,b) for a,b in ps]
    return list(zip(*ps))
def electrodegratingft(barstarts,barends):
    a,b = np.array(barstarts),np.array(barends)
    w = 1-grating2wave(a,b)
    # w.plot()
    Λ = 0.5*np.mean(a[1:]-a[:-1]) + 0.5*np.mean(b[1:]-b[:-1])
    L = b[-1]-a[0]
    dΛ = 10*Λ*Λ/L
    vx = np.linspace(Λ-dΛ,Λ+dΛ,501)
    def waveft(w):
        return Wave(abs(w.ft(1/vx,norm=True)),vx)
    w0 = grating2wave(*grating(Λ,0.5,L,padcount=1,gapx=0))
    us = [waveft(u) for u in (w0,w)]
    Wave.plots(*[u/us[0](Λ) for u in us],x='Λ (µm)',y='relative nonlinear response')
def onoffs2grating(d,pattern,offvalue=0,δ=0,L=None,wave=False):
    pattern = [(1 if p in '1*#^x+' else offvalue) for p in pattern] if isinstance(pattern, str) else list(pattern)
    L = L if L is not None else d*len(pattern)
    N = int(L//(d*len(pattern)))
    ps = [0] + N*pattern + [0]
    xys = [(n*d+f*δ,p) for n,(p0,p1) in enumerate(zip(ps[:-1],ps[1:])) for p,f in ((p0,-1),(p1,+1)) if p0!=p1]
    a,b = bool(0<xys[0][0]), bool(xys[-1][0]<L)
    xys = a*[(0,offvalue)] + xys + b*[(L,offvalue)]
    return Wave.fromxys(xys) if wave else gratingfromxys(xys)
def deltasigmamatch(target):
    p = np.arange(len(target)) % 2 # parity mask [0 1 0 1 0 ...]
    return np.round((target - p) / 2) * 2 + p # snap to the nearest integer with the required parity
def fixedbarapodized(Λ,gx,σ,wave=False): # Λ in µm, gx in µm
    from scipy.special import erf
    Λ = abs(Λ)
    def g(n): # half gauss, n = halfbar index
        return np.exp(-0.5*n**2/σn**2)
    def intg(n): # integral of half gauss
        return σn * np.sqrt(np.pi/2) * erf( n / (np.sqrt(2)*σn) )
    N,σn = int(gx/Λ-1),σ*2*gx/Λ # N = number of halfbars in last half of grating, not counting the middle halfbar
    ns = [0]+[0.5+i for i in range(N)]
    # t = g(Wave(ns,ns)).trapezoid().rename('t')
    h = Wave([intg(n) for n in ns],ns,'h')
    # mh = Wave(deltasigmamatch(h.y[1:]-0.5)+0.5, h.x[1:])
    mt = Wave(deltasigmamatch(h.y[1:]-0.5)+0.5, h.x[1:])
    # Wave.plot(t,h,mt,mh,l='2323',grid=1,m='o',ms=3,lw=1.5,seed=1)
    d = mt.diff() # ±1 for each halfbar, +1=in-phase, -1=out-of-phase
    assert all(di in (-1,+1) for di in d.y)
    flip = [int(f) for f in ~(d.y==+1)^(np.arange(len(d))%2)] # e.g. flip=[0 1 0 1...] if d is all +1 or [1 0 1 0...] if d is all -1
    ff = flip[::-1] + [1] + flip # flips is for second half, ff is for full length including middle halfbar
    ff = [ff[0]^fi for fi in ff] # invert if needed to ensure beginning and end are unpoled
    assert ff==ff[::-1], 'flipping pattern must be symmetric'
    return onoffs2grating(Λ/2,ff,wave=wave)
def apodizedgrating(period,gx,f=1,overfill=False): # f = apodization function or constant 0<f≤1 relative amplitude
    func = (lambda x:f) if not callable(f) else f
    n = int(gx/period)
    n = n - (n+1)%2 + (2 if overfill else 0) # number of bars, odd for symmetry, central bar centered
    xs = period * np.arange(1.*n) + (gx-(n-1)*period)/2 # centers of each bar, bar n//2 is at exactly L/2
    barstrengths = func(xs/gx)
    barlengths = period * np.arcsin(barstrengths)/pi # relative strength of a bar given its duty cycle dc is sin(pi*dc)
    barstarts,barends = xs - 0.5*barlengths, xs + 0.5*barlengths
    return barstarts,barends
def gaussapodizedgrating(period,gx,σ=0.23,A=1): # 0<A≤1 relative amplitude
    def a(x): return A * np.exp( -0.5 * (x-0.5)**2  / σ**2 ) # 0 to 1 to 0
    return apodizedgrating(period,gx,f=a)
def bestphase(starts,ends,Λ): # return phase in µm to shift grating to maximize overlap with cos(2πx/Λ)
    c = sum(+sin(2*pi*b/Λ)-sin(2*pi*a/Λ) for a,b in zip(starts,ends))
    s = sum(-cos(2*pi*b/Λ)+cos(2*pi*a/Λ) for a,b in zip(starts,ends))
    φ = np.atan2(s,c)
    φ = φ + 2*pi if φ<0 else φ
    return φ/(2*pi)*Λ # bestphase([1],[3],4)==2, bestphase([0],[2],4)==1
def splitbarsalongperiodboundaries(starts,ends,Λ,x0): # x0 is phase, or the center of a period
    def periodbinindex(x):
        return int((x-x0+0.5*Λ)//Λ)
    newstarts,newends = [],[]
    for a,b in zip(starts,ends):
        i,j = periodbinindex(a),periodbinindex(b)
        xs = [a] + [x0 + (k-0.5)*Λ for k in range(i+1,j+1)] + [b]
        newstarts,newends = newstarts + xs[:-1], newends + xs[1:]
    return newstarts,newends
def gratingamplitude(starts,ends,Λ,smooth=0): # each point is the average strength over one period with max of 1
    if hasattr(Λ,'__iter__'):
        assert not isinstance(Λ,str)
        return [gratingamplitude(starts,ends,x,smooth=smooth) for x in Λ]
    from collections import defaultdict
    x0 = bestphase(starts,ends,Λ)
    n = int((ends[-1]-starts[0])//Λ)
    n = n + 7 + n%2 # an odd number of periods to more than cover L
    m = int(round((0.5*(ends[-1]+starts[0])-x0)/Λ))
    c0 = x0 + m*Λ # find c0 that is closest to bars' center and is of the form x0 + m*Λ, c0 is the center of the central period
    periodcenters = [c0 + (i-n//2)*Λ for i in range(n)]
    periodboundaries = [c0 + (i-n//2-0.5)*Λ for i in range(n+1)]
    def periodbinindex(x): return int((x-x0+0.5*Λ)//Λ)
    newstarts,newends = splitbarsalongperiodboundaries(starts,ends,Λ,c0)
    periodbars = defaultdict(list) # bin the new bars into their periods
    for a,b in zip(newstarts,newends):
        periodbars[periodbinindex(0.5*a+0.5*b)] += [(a,b)]
    def onshare(a,b): return (sin(2*pi*(b-x0)/Λ) - sin(2*pi*(a-x0)/Λ)) / (2*pi) * (pi/2) # onshare is the positive contribution to grating amplitude, multiply by pi/2 to normalize to 1 for the max response of one period
    shares = [sum(2*onshare(a,b) for a,b in periodbars[m + i - n//2]) for i in range(n)] # since the contribution of a whole poled or unpoled period is zero, the contribution of the gaps is equal to the contribution of the bars, hence factor of 2
    return Wave(shares,periodcenters) if smooth==0 else Wave(shares,periodcenters).smooth(smooth)


class Grating: # bars defined by starts,ends; transforms return new instances, chainable
    # methods intentionally shadow the module-level functions of the same name; inside a
    # method body the bare name resolves to the module function (class scope isn't searched)
    def __init__(self, starts, ends, length=None, validate=True): # validate=False allows zero-width or touching bars mid-pipeline (e.g. after expandbars, before dropsmallbars)
        self.starts, self.ends = list(starts), list(ends)
        self.length = length # device length; bars exceeding [0,length] are unconventional but not invalid, test with isinbounds()
        assert len(self.starts)==len(self.ends), 'starts and ends must have equal length'
        if validate:
            assert self.isvalid(), 'bars must be strictly ordered: starts[n] < ends[n] < starts[n+1]'
    def isvalid(self):
        bs = self.bars
        return all(b0<b1 for b0,b1 in zip(bs[:-1],bs[1:]))
    def validate(self):
        assert self.isvalid(), 'bars must be strictly ordered: starts[n] < ends[n] < starts[n+1]'
        return self
    def isinbounds(self): # True if all bars lie within [0,length], inclusive; vacuously True if length is None or no bars
        if self.length is None or not self.starts:
            return True
        return 0<=min(self.starts) and max(self.ends)<=self.length
    @property
    def barstarts(self): return self.starts
    @property
    def barends(self): return self.ends
    @property
    def bars(self): # [s0,e0,s1,e1,...]
        return grating2bars(self.starts,self.ends)
    @classmethod
    def frombars(cls, bs, length=None):
        return cls(*bars2grating(bs),length=length)
    @property
    def period(self): # mean period, same estimate as ftgrating's default p0
        a,b = np.array(self.starts),np.array(self.ends)
        return float( np.diff(a).mean()/2 + np.diff(b).mean()/2 )
    ## dunders
    def __iter__(self): # starts,ends = g, also f(*g) for module functions
        return iter((self.starts,self.ends))
    def __len__(self): # number of bars (note: not len(list(self)), which is 2)
        return len(self.starts)
    def __getitem__(self,i): # g[i] = (start,end) of bar i, g[i:j] = Grating slice
        if isinstance(i,slice):
            return Grating(self.starts[i],self.ends[i],length=self.length)
        return (self.starts[i],self.ends[i])
    def __eq__(self,other): # compares bars only, not length
        try:
            (a0,b0),(a1,b1) = self,other
        except (TypeError,ValueError):
            return NotImplemented
        return len(a0)==len(a1) and np.allclose(a0,a1) and np.allclose(b0,b1)
    def __and__(self,other): # g & h = binaryand
        return self.binaryand(other)
    def __or__(self,other): # g | h = binaryor
        return self.binaryor(other)
    def __repr__(self):
        def f(xs):
            return '['+','.join(f'{x:g}' for x in xs)+']' if len(xs)<7 else f'[{xs[0]:g},{xs[1]:g},...,{xs[-1]:g}]'
        s = f'Grating({f(self.starts)},{f(self.ends)}'
        return s + (f',length={self.length:g})' if self.length is not None else ')')
    ## transforms, return new Grating (length is inherited)
    # def mergetouchingbars(self,tolerance=0.0):
    #     return Grating(*mergetouchingbars(*self,tolerance=tolerance),length=self.length)
    # def dropsmallbars(self,tolerance=0.0):
    #     return Grating(*dropsmallbars(*self,tolerance=tolerance),length=self.length)
    def mergetouchingbars(self,tolerance=0.0,validate=True): # validate=False: won't create invalid bars, but can still pass them
        return Grating(*mergetouchingbars(*self,tolerance=tolerance),length=self.length,validate=validate)
    def dropsmallbars(self,tolerance=0.0,validate=True):
        return Grating(*dropsmallbars(*self,tolerance=tolerance),length=self.length,validate=validate)
    def expandbars(self,op): # may create zero-width or overlapping bars, chain .dropsmallbars().mergetouchingbars()
        return Grating(*expandbars(*self,op),length=self.length,validate=False)
    def shrinkbars(self,dx):
        return self.expandbars(-dx)
    def invertbarsgaps(self):
        return Grating(*invertbarsgaps(*self),length=self.length)
    def apodizebars(self,afunc=None,apodize=None,σ=0.23): # may create touching bars, chain .dropsmallbars(tolerance=0)
        return Grating(*apodizebars(*self,afunc=afunc,apodize=apodize,σ=σ),length=self.length,validate=False)
    def breakupbars(self,maxbar,gapsize):
        return Grating(*breakupbars(*self,maxbar=maxbar,gapsize=gapsize),length=self.length)
    def breakupgaps(self,maxgap,barsize):
        return Grating(*breakupgaps(*self,maxgap=maxgap,barsize=barsize),length=self.length)
    def splitbarsalongperiodboundaries(self,Λ,x0): # split bars touch at period boundaries by construction
        return Grating(*splitbarsalongperiodboundaries(*self,Λ,x0),length=self.length,validate=False)
    def spectralcomb(self,offfraction,L=None): # L defaults to self.length, then to last bar end
        L = L if L is not None else self.length
        return Grating(*spectralcomb(*self,offfraction,L=L),length=self.length)
    def piphasegrating(self):
        return Grating(*piphasegrating(*self),length=self.length)
    def phasegrating(self,n,x0=0,x1=None):
        return Grating(*phasegrating(*self,n,x0=x0,x1=x1),length=self.length)
    def reverse(self,L=None): # L defaults to self.length
        L = L if L is not None else self.length
        assert L is not None, 'L required if length not set'
        return Grating(*reverse(L,*self),length=self.length)
    def binaryand(self,other): # length taken from self (left operand)
        return Grating(*binaryand(tuple(self),tuple(other)),length=self.length)
    def binaryor(self,other): # length taken from self (left operand)
        return Grating(*binaryor(tuple(self),tuple(other)),length=self.length)
    def alternatinggrating2(self,other,L=None,repeats=1): # L defaults to self.length
        L = L if L is not None else self.length
        assert L is not None, 'L required if length not set'
        return Grating(*alternatinggrating2(tuple(self),tuple(other),L,repeats=repeats),length=self.length)
    ## computed, return Wave/float/xy
    def grating2xy(self,delta=0):
        return grating2xy(*self,delta=delta)
    def grating2wave(self,delta=0):
        return grating2wave(*self,delta=delta)
    def gratingperiod2wave(self):
        return gratingperiod2wave(*self)
    def ftgrating(self,p0=None,dp=None,normalize=True,amplitude=True,res=2001):
        return ftgrating(*self,p0=p0,dp=dp,normalize=normalize,amplitude=amplitude,res=res)
    def bestphase(self,Λ):
        return bestphase(*self,Λ)
    def gratingamplitude(self,Λ=None,**kwargs):
        return gratingamplitude(*self,Λ if Λ is not None else self.Λ,**kwargs)
    def electrodegratingft(self):
        return electrodegratingft(*self)
    def bars2file(self,file):
        bars2file(file,self.bars)
        return self
    def plot(self,**kwargs):
        return self.grating2wave().plot(**kwargs)
    def save(self,name,file):
        def list2str(a,f='{:g}',sep=','): # e.g. print(list2str([1,2,3],f='{:.1f}',sep='#')) # 1.0#2.0#3.0
            if 0==len(a): return '[]'
            if hasattr(a[0],'__len__') and not isinstance(a[0],str):
                aa = [f"({list2str(ai,f,sep=',')})" for ai in a]
                return sep.join((len(aa)*['{}'])).format(*aa)
            return '[' + sep.join((len(a)*[f])).format(*a) + ']'
        with open(file,'a') as f:
            f.write(f'barstarts{name},barends{name} = {list2str(self.barstarts)},{list2str(self.barends)}\n')

class Uniformgrating(Grating): # Grating that remembers its construction parameters, wraps grating()
    # transforms and slices return plain Gratings: a transformed uniform grating is no longer parametrically uniform
    def __init__(self, period, dc=0.5, length=None, gapx=0, phasex=0, x0=0, apodize=None, omitpads=()):
        assert length is not None, 'length required'
        self.Λ, self.dc = period, dc
        self.gapx, self.phasex, self.x0, self.apodize, self.omitpads = gapx, phasex, x0, apodize, omitpads
        super().__init__(*grating(period,dc,padx=length,padcount=1,gapx=gapx,phasex=phasex,x0=x0,apodize=apodize,omitpads=omitpads),length=length)
    @property
    def period(self): # exact design period (base class estimates it from the bars)
        return self.Λ
    def __repr__(self):
        defaults = dict(gapx=0,phasex=0,x0=0,apodize=None,omitpads=())
        s = ','.join(f'{k}={getattr(self,k)!r}' for k,d in defaults.items() if getattr(self,k)!=d)
        head = f'Uniformgrating({self.Λ:g},' + (f'{self.dc:g},' if self.dc!=0.5 else '') + f'length={self.length:g}'
        return head + (','+s if s else '') + ')'
class Chirpedgrating(Grating): # wraps kchirpgrating()
    def __init__(self, p0, p1, dc=0.5, length=None, gapx=0):
        assert length is not None, 'length required'
        self.p0, self.p1, self.dc, self.gapx = p0, p1, dc, gapx
        super().__init__(*kchirpgrating(p0,p1,dc,padx=length,padcount=1,gapx=gapx),length=length)
    @property
    def period(self): # center-k design period = harmonic mean of p0,p1 = grating length/barcount
        return 2*self.p0*self.p1/(self.p0+self.p1)
    def __repr__(self):
        head = f'Chirpedgrating({self.p0:g},{self.p1:g},' + (f'{self.dc:g},' if self.dc!=0.5 else '') + f'length={self.length:g}'
        return head + (f',gapx={self.gapx:g}' if self.gapx else '') + ')'
class Dualperiodgrating(Grating):
    @property
    def periods(self):
        return (self.period1,self.period2)
    def gratingamplitude(self,Λ=None,**kwargs):
        Λ = Λ if Λ is not None else self.periods
        return gratingamplitude(*self,Λ,**kwargs)
class Interleavedgrating(Dualperiodgrating): # wraps geometricinterleavedgrating()
    def __init__(self, period1, period2, length=None, φ1=None, φ2=None):
        assert length is not None, 'length required'
        self.period1, self.period2 = period1, period2
        self.φ1 = -length*pi/period1 if φ1 is None else φ1 # resolved defaults, bars symmetric about length/2
        self.φ2 = -length*pi/period2 if φ2 is None else φ2
        super().__init__(*geometricinterleavedgrating(period1,period2,length,φ1=self.φ1,φ2=self.φ2),length=length)
    def __repr__(self):
        s = ','.join(f'{k}={v:g}' for k,v,d in [('φ1',self.φ1,-self.length*pi/self.period1),('φ2',self.φ2,-self.length*pi/self.period2)] if v!=d)
        return f'Interleavedgrating({self.period1:g},{self.period2:g},length={self.length:g}' + (','+s if s else '') + ')'
class Legacyinterleavedelectrode(Dualperiodgrating): # interleaved electrode bars as built by the legacy mask code (addinterleavedsubmount)
    # note: interleavedgrating is deliberately called with (period1,period0), preserving the legacy argument order
    def __init__(self, period1, period2, padcount=8, gx=2500, smallestbar=1, overpole=0, breakupgapsize=0, apodize=None):
        self.period1, self.period2, self.padcount, self.gx = period1, period2, padcount, gx
        self.smallestbar, self.overpole, self.breakupgapsize, self.apodize = smallestbar, overpole, breakupgapsize, apodize
        g0 = interleavedgrating(period2,period1,padcount,gx) if apodize is None else apodizedinterleavedgrating(period2,period1,padcount,gx,apodize=apodize)
        g = ( Grating(*g0,length=padcount*gx)
              .shrinkbars(overpole)
              .mergetouchingbars(tolerance=smallestbar,validate=False)
              .dropsmallbars(tolerance=smallestbar)
              .breakupgaps(maxgap=breakupgapsize,barsize=smallestbar)
              .validate() )
        super().__init__(*g,length=padcount*gx)
    def __repr__(self):
        defaults = dict(padcount=8,gx=2500,smallestbar=1,overpole=0,breakupgapsize=0,apodize=None)
        s = ','.join(f'{k}={getattr(self,k)!r}' for k,d in defaults.items() if getattr(self,k)!=d)
        return f'Legacyinterleavedelectrode({self.period1:g},{self.period2:g}' + (','+s if s else '') + ')'
class Discreteinterleavedelectrode(Dualperiodgrating): # discreteapodizedinterleavedgrating
    def __init__(self, period1, period2, xeff=None, A1=None, A2=None, padcount=8, gx=2500, smallestbar=1, overpole=0, res=0.1, σ0=0.23):
        x0 = padcount*gx
        xeff = x0 if xeff is None else xeff
        σ = σ0*xeff/x0
        self.period1, self.period2, self.xeff, self.A1, self.A2 = period1, period2, xeff, A1, A2
        self.padcount, self.gx, self.smallestbar, self.overpole, self.res, self.σ0 = padcount, gx, smallestbar, overpole, res, σ0
        g0 = discreteapodizedinterleavedgrating(period1,period2,x0,σ=σ,A1=A1,A2=A2,res=res)
        g = Grating(*g0,length=x0).shrinkbars(overpole).mergetouchingbars(tolerance=smallestbar,validate=0).dropsmallbars(tolerance=smallestbar).validate()
        super().__init__(*g,length=x0)
class Geometricinterleavedelectrode(Dualperiodgrating):
    def __init__(self, period1, period2, xeff=None, A1=None, A2=None, padcount=8, gx=2500, smallestbar=1, overpole=0, σ0=0.23):
        x0 = padcount*gx
        xeff = x0 if xeff is None else xeff
        σ = σ0*xeff/x0
        self.period1, self.period2, self.xeff, self.A1, self.A2 = period1, period2, xeff, A1, A2
        self.padcount, self.gx, self.smallestbar, self.overpole, self.σ0 = padcount, gx, smallestbar, overpole, σ0
        g0 = geometricapodizedinterleavedgrating(period1,period2,x0,f1=A1,f2=A2,σ=σ)
        g = Grating(*g0,length=x0).shrinkbars(overpole).mergetouchingbars(tolerance=smallestbar,validate=0).dropsmallbars(tolerance=smallestbar).validate()
        super().__init__(*g,length=x0)

def classtests():
    g = Grating([0,2,4],[1,3,5])
    starts,ends = g
    assert starts==[0,2,4] and ends==[1,3,5]
    assert len(g)==3 and g[1]==(2,3) and g[1:]==Grating([2,4],[3,5])
    assert g==Grating.frombars([0,1,2,3,4,5]) and g.bars==[0,1,2,3,4,5]
    assert np.isclose(g.period,2)
    assert g.invertbarsgaps()==Grating([1,3],[2,4])
    assert g.expandbars(1)==Grating([-0.5,1.5,3.5],[1.5,3.5,5.5],validate=False)
    assert g.expandbars(1).mergetouchingbars(tolerance=0)==Grating([-0.5],[5.5])
    assert g.shrinkbars(0.5)==Grating([0.25,2.25,4.25],[0.75,2.75,4.75])
    assert g.shrinkbars(1).dropsmallbars(tolerance=0)==Grating([],[])
    assert g.reverse(6)==Grating([1,3,5],[2,4,6])
    assert (g & Grating([0.5],[10]))==Grating([0.5,2,4],[1,3,5])
    assert (g | Grating([0.9],[2.1]))==Grating([0,4],[3,5])
    assert Grating([0,2,4,6],[1,3,5,7]).spectralcomb(0.1,L=8)==Grating([0,2,4.4,6],[1,3,5,7])
    assert np.isclose(Grating([1],[3]).bestphase(4),2) and np.isclose(Grating([0],[2]).bestphase(4),1)
    assert Grating([0],[10]).breakupbars(4,1).isvalid() and 3==len(Grating([0],[10]).breakupbars(4,1))
    g100 = Grating(*grating(100,0.5,padx=2000))
    assert g100.phasegrating(2).isvalid() and g100.piphasegrating().isvalid()
    assert g100.apodizebars(apodize='trapezoidal').dropsmallbars(tolerance=0)==Grating(*grating(100,0.5,2000,apodize='trapezoidal'))
    h = g100.splitbarsalongperiodboundaries(100,50)
    assert np.isclose(sum(b-a for a,b in zip(*h)),sum(b-a for a,b in zip(*g100))) # poled area preserved
    ## length tests
    assert Grating([],[],length=1000).length==1000 and Grating([],[],length=1000).isinbounds()
    gl = Grating([0,2,4],[1,3,5],length=6)
    assert gl.length==6 and gl.invertbarsgaps().length==6 and gl[1:].length==6 and gl.reverse().length==6
    assert gl.reverse()==Grating([1,3,5],[2,4,6],length=6) # reverse L defaults to length
    assert (gl & Grating([0.5],[10])).length==6
    assert gl.isinbounds() and Grating([0],[6],length=6).isinbounds() # bounds are inclusive
    assert not Grating([0],[7],length=6).isinbounds() and not Grating([-1],[3],length=6).isinbounds() # out of bounds is constructible, merely unconventional
    assert not gl.expandbars(1).mergetouchingbars(tolerance=0).isinbounds()
    assert Grating([0],[7]).isinbounds() # no length, nothing to violate
    ## Uniformgrating tests
    u = Uniformgrating(100,length=2000)
    assert u==Grating(*grating(100,0.5,2000)) and u.period==100 and u.dc==0.5 and u.length==2000 and u.isinbounds()
    assert Uniformgrating(100,0.5,2000)==u # length is third positional
    assert Uniformgrating(100,length=2000,apodize='trapezoidal')==Grating(*grating(100,0.5,2000,apodize='trapezoidal'))
    assert not Uniformgrating(100,length=2000,apodize='trapezoidal').isinbounds() # apodization widens edge bars about fixed centers, past the pad
    assert not Uniformgrating(100,length=20001).isinbounds() # ceil barcount, last bar ends at 20050
    assert isinstance(u.shrinkbars(1),Grating) and not isinstance(u.shrinkbars(1),Uniformgrating) # transforms decay to Grating
    assert isinstance(u[1:],Grating) and not isinstance(u[1:],Uniformgrating)
    assert u.reverse().isvalid() and u.piphasegrating().isvalid()
    assert repr(u)=='Uniformgrating(100,length=2000)'
    assert repr(Uniformgrating(100,0.6,2000,x0=50))=='Uniformgrating(100,0.6,length=2000,x0=50)'
    ## Chirpedgrating, Interleavedgrating tests
    c = Chirpedgrating(14.45,14.55,length=51000)
    assert c==Grating(*kchirpgrating(14.45,14.55,0.5,51000)) and c.length==51000 and c.p0==14.45
    assert np.isclose(c.period, 2*14.45*14.55/(14.45+14.55)) # center-k period, harmonic mean
    assert abs(Grating(*c).period - c.period)<0.01 # agrees with the base class bar-based estimate
    assert isinstance(c.dropsmallbars(),Grating) and not isinstance(c.dropsmallbars(),Chirpedgrating)
    assert repr(c)=='Chirpedgrating(14.45,14.55,length=51000)'
    ii = Interleavedgrating(7,17,length=1000)
    assert ii==Grating(*geometricinterleavedgrating(7,17,1000)) and ii.periods==(7,17) and ii.length==1000
    assert np.isclose(Interleavedgrating(7,17,1000,φ1=0).φ1,0) and np.isclose(ii.φ1,-1000*pi/7)
    assert isinstance(ii[1:],Grating) and not isinstance(ii[1:],Interleavedgrating)
    assert repr(ii)=='Interleavedgrating(7,17,length=1000)'
    assert repr(Interleavedgrating(7,17,1000,φ1=0))=='Interleavedgrating(7,17,length=1000,φ1=0)'
    ## validate kwarg on cleanup methods (overpole phantoms are benign mid-pipeline)
    d = Grating([0,3.5,6],[2,3.7,8]).shrinkbars(0.6) # middle bar becomes a phantom (negative width)
    try:
        d.mergetouchingbars(tolerance=1); assert False, 'should have raised'
    except AssertionError: pass
    e = d.mergetouchingbars(tolerance=1,validate=False).dropsmallbars(tolerance=1)
    assert e==Grating([0.3,6.3],[1.7,7.7])
    assert np.isclose(e.starts[1]-e.ends[0],(6-2)+0.6) # final gap = original separation + overpole
    print('classtests passed')

if __name__ == '__main__':
    classtests()
    def simplephasegrating():
        g0,g1 = grating(100,0.5,padx=20001,padcount=1,gapx=0,phasex=0,x0=0)
        n = len(g1)//2
        a0,a1 = g0[:n],g1[:n]
        b0,b1 = g0[n:],g1[n:]
        b0,b1 = invertbarsgaps(b0,b1)
        # b0,b1 = list(np.array(b0)+50),list(np.array(b1)+50)
        f = ftgrating(g0,g1,p0=100,dp=6,normalize=0,res=2001)**2
        ff = ftgrating(a0+b0,a1+b1,p0=100,dp=6,normalize=0,res=2001)**2
        norm = f.max()
        Wave.plots(f/norm,ff/norm)
    simplephasegrating()
