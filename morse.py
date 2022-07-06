#EPROM state machines for the encoding and decoding of morse code signals
#There are three state machines, plus a few lookup tables:
# State machines: 
#  1. Morse receiver (Morse serial in, state out)
#  2. Serial transmitter (Morse state from Morse receiver in, 8bit serial out, odd parity)
#  3. Morse Transmitter (start state in, Morse serial out)
#
#Lookup tables:
#  1. Morse receiver state in, ASCII out
#  2. ASCII in, Morse receiver state out
#  3. ASCII in, Morse Transmitter start state out
#
#Tere is also a "Logisim" simulation "morse.circ" that uses the ROMs generatet by this Python program.

#valid morse codes taken from "https://morsecode.world/international/translator.html"
#for all morse codes of length up to 6 dashes or dots
#input for the translator generated as follows:
# ' '.join(bin2dashdot(i) for i in range(128))
#characters '^' and ' ' prepended to the result for code 0b0 and 0b1 
morse={character.lower():i for i,character in enumerate(
  '^ '+'ETIANMSURWDKGOHVFÜLÄPJBXCYZQÖŠ54Ŝ3É#Ð2&È+#ÞÅĴ16=/#Ç#(#7ŻĜÑ8#90############?#####"##.####@###\'##-#########!#)####Ź,####:#######')}

#define some undefined morse codes to fill gaps in the binary trees:
  
morse.update({'$':0b1_10101, '°':0b1_01011, '#':0b1_000111, '*':0b1_00101, '[':0b1_001011, ']':0b1_010011,'{':0b1_000001,'}':0b1_011000, 
'þ':0b1_01100, '<':0b1_10111,'\\':0b1_10011,})

#The binary encoding is: 0='.', and 1:'-', with a '1' prepended as length indicator.
#e.g. the leter 'e' is 0b1_0 (a single dot), and 't' is 0b1_1 (a single dash)
#The letter 'a' is 0b1_01 (a dot followed by a dash), etc.
#The binary number is also the number ot the node in the binary tree:
#
#                    0b0='^'
#                      |
#                      |
#                    0b1=' '
#                  /        \
#                /           \
#         0b1_0='e'          0b1_1='t'
#         /     \            /       \
#  0b1_00='i' 0b1_01='a'  0b1_10='n' 0b1_11='m' 
#    /   \      /  \       /   \      /   \
# ...

#The binary tree is then mapped to a 6-dimensional hypercube that is used in the state machines.
#For the receiver, the state starts at the root, and moves through the nodes of the tree towards the leaves,
#branching left for a received dot, and right for a dash:

receivercube=[['y(#ð', '\\"è?','x/l&', "6-1'"],
              ['k)3f', 'nt e', 'dqra', 'b=jw'],
              ['c$vu', 'çmsi', '.gä+', '7zżp'],
              ['9!2ü', 'šoh5', '0ö4@', ':8,å'],]  

#The transmitter state machine starts at an initial state corresponing to the character to be transmitted, 
#and the state works its way through the nodes of the tree towards the root.
#The state changes synchronously with a clock input to the state machine. The appropriate Morse signal level
#is generated as output for each state as the state moves towards the root of the tree.

transmittercube=[['.xk0', '$*äq', '5hsi', '6ldn'], 
                 ['-vua', '{4^t', '&b e', '7zgr'], 
                 ['<2jm', '°=ow', '[8+c', '}þpf'], 
                 [',]š1', 'y\\3ü','!9ĵ/', ':?öŝ']]


def count_bits(n):
  n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
  n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
  n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
  n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
  n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
  n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32) # This last & isn't strictly necessary.
  return n
  
def parity(i):
  return (count_bits(i)&1)==0

def dashdot2bin(dashdot,*,dd2bin=''.maketrans('.-_','011')):
  return int(((f'1{dashdot}').translate(dd2bin)),2)
  
def bin2dashdot(bincode,*,bin2dd=''.maketrans('01','.-')):
  return (f'{bincode:b}')[1:].translate(bin2dd)
  
class Node:    
  def __init__(self,id,left=None,right=None,maxdepth=20):
    self._id=id
    self._parent=None
    self._children=[None, None]
    self._setchild(0,left)
    self._setchild(1,right)
    self._maxdepth=maxdepth
  def __repr__(self):
    return f'Node({self._id})' if self._maxdepth==20 else f'Node({self._id},maxdepth={self._maxdepth})'
  @property
  def id(self):
    return self._id
  @property
  def isleftchild(self):
    return (self._parent!=None) and (self._parent.left is self)
  @property
  def isrightchild(self):
    return (self._parent!=None) and (self._parent.right is self)
  @property
  def isroot(self):
    return self._parent==None
  @property
  def parent(self):
    return self._parent
  def _setchild(self,i,value):
    if self._children[i] !=None:
      self._children[i]._parent=None
    if value!=None:
      value._parent=self
    self._children[i]=value
    return
  @property
  def left(self):
    return self._children[0]
  @left.setter
  def left(self,value):
    self._setchild(0,value)
    return
  @property
  def right(self):
    return self._children[1]
  @right.setter
  def right(self,value):
    self._setchild(1,value)
    return
  @property
  def root(self):
    parent=self
    for i in range(self._maxdepth):
      if parent.parent==None:
        return parent
      parent=parent.parent
    raise Exception(f'Maximum tree depth exceeded! ({self.__repr__})')
  @property
  def children(self,includeNone=False):
    for child in self._children:
      if (child!=None) or includeNone:
        yield child
    return
  def nodes(self,includeNone=False,maxdepth=None):
    if maxdepth==None: maxdepth=self._maxdepth
    if maxdepth==0:
      return
    yield self
    for child in self._children:
      if child!=None:
        yield from child.nodes(includeNone=includeNone,maxdepth=maxdepth-1)
      elif includeNone:
        yield from (None for _ in range((1<<(maxdepth-1))-1))
  def edges(self,includeNone=None,maxdepth=None,reverse=False):
    nodes=self.nodes(includeNone=includeNone,maxdepth=maxdepth)
    next(nodes)#skip self
    for node in nodes:
      edgetype=0
      if node.isrightchild:
        edgetype=1 
      if node.parent.id==0:
        edgetype=2
      if reverse:
        yield (node, node.parent, edgetype,)
      else:
        yield (node.parent, node, edgetype,)
  @property
  def depth(self):
    return 1+max(child.depth if child!=None else 0 for child in self._children)
  @property
  def level(self):
    if self.parent==None:
      return 0
    else:
      return 1+self.parent.level

def reversemorse(b):
  if b==0:
    return 0
  a=1
  while b>1:
    a<<=1
    a|=b&1
    b>>=1
  return a
  
def maketree(cube,treeids,reverse=False):       
  treenodes=tuple(Node(i) for i in range(128))
  treeNodeForLabel={}
  cubenodes=[[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
  treeNodeForCubeId=[None for _ in range(64)]
  for z,plane in enumerate(cube):
    for y,row in enumerate(plane):
       for x,c in enumerate(row):
         treeid=treeids[c]
         if reverse:
           treeid=reversemorse(treeid)
         treenode=treenodes[treeid]
         treenode.reverse=reverse
         treenode.label=c   
         treeNodeForLabel[c]=treenode
         treenode.nodeForLabel=treeNodeForLabel
         msb=treeid.bit_length()
         if msb==0:
           x_=0.5
           y_=-1
         else:
           m=(1<<msb)>>1
           x_=(treeid+0.5)/m-1.0 if msb!=0 else 0.5
           y_=-(msb)
         if reverse:
           y_+=-count_bits(treeid)
         treenode.coord=(x_,y_)
         cubeid= ((z^(z>>1)^1)<<4) | ((y^(y>>1)^1)<<2) | ((x^(x>>1)^3)<<0) 
         treenode.cubeid=cubeid
         treenode.cubecoord=(x-1.5,-(y-1.5),-(z-1.5))
         treenode.cubeindex=(x,3-y,3-z,)
         cubenodes[x][3-y][3-z]=treenode
         treenode.cubenodes=cubenodes
         treeNodeForCubeId[cubeid]=treenode
         treenode.nodeForCubeId=treeNodeForCubeId
         if reverse:
           if (treeid&1)==1:
             parentid=treeid^1
             right=1
           else: 
             parentid,right=divmod(treeid,2)
         else:
             parentid,right=divmod(treeid,2)
         if right==1:
           treenodes[parentid].right=treenode
         elif parentid!=treeid:
           treenodes[parentid].left=treenode
  rootnode=cubenodes[2][2][2]
  if rootnode.parent!=None:
    if rootnode.parent.right is rootnode:
      rootnode.parent.right=None
    else:
      rootnode.parent.left=None
  return rootnode
   
receivertree=maketree(receivercube,morse,reverse=False)    
transmittertree=maketree(transmittercube,morse,reverse=True)    

def scale(points,scale=(1.0,1.0)):
  return list([xi*si for xi,si in zip(p,scale)] for p in points)
  
def arrow_lines(start,end, head_pos=1, head_anchor=-1.0, head_length=0.1, head_width=0.06):
    s=complex(*start)
    e=complex(*end)
    v=e-s
    l=abs(v)
    if l==0:
      return
    ev=v/l
    p_h=complex(-head_length,0.5*head_width)
    p_tip=s+v*head_pos+ev*head_length*head_anchor
    arrow_points=(s,p_tip,p_tip+p_h*ev,p_tip+p_h.conjugate()*ev,p_tip,e,)
    return [(p.real,p.imag,) for p in arrow_points]
  
  
def plotTree(tree,width=7,height=4.5):
  from matplotlib import pyplot as plt
  from matplotlib.collections import LineCollection
  plt.close()
  fig = plt.figure()
  fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
  ax = fig.add_subplot(111)
  fig.set_size_inches(width,height)
  xscale=width
  yscale=height/tree.depth
  treeEdges=LineCollection([arrow_lines(*scale((node.coord for node in edge[:2]),scale=(xscale,yscale,))) for edge in tree.edges(reverse=tree.reverse)],colors=[['r','b','k'][edge[2]] for edge in tree.edges(reverse=tree.reverse)],linewidth=1.5,zorder=0)
  ax.add_collection(treeEdges)
  ax.scatter(*zip(*(scale((node.coord,), scale=(xscale,yscale,))[0] for node in tree.nodes())), color='w',marker='o',s=180,edgecolors=['lime' if node.level%2==0 else 'magenta'for node in tree.nodes() for x,y in (node.coord,)],zorder=100)
  for node in tree.nodes():
    c=node.label
    special_characters={'ż':r'$\dot{\rm{z}}$', 'ĝ':r'$\hat{\rm{g}}$','ŝ':r'$\hat{\rm{s}}$', 
      'ĵ':r'$\hat{\rm{j}}$', ' ':r'_'}
    c=special_characters.get(c,c)
    ax.text(*(scale((node.coord,),scale=(xscale,yscale,))[0]),c, size=10, zorder=700,  color='k',ha='center',va='center')
  plt.ylim(-height-0.5*height/tree.depth,-0.5*height/tree.depth)
  plt.xlim(0,width)
  plt.axis('off')
  plt.show()

import numpy as np
from math import sin,cos,pi

def projectionMatrix(rz,rx):
  Mz=[[cos(rz),sin(rz),0],[-sin(rz),cos(rz),0],[0,0,1]]
  Mx=[[1,0,0],[0,cos(rx),sin(rx)],[0,-sin(rx),cos(rx)]]
  return np.dot(Mz,Mx)
  
def arrow3Dsegments(start,end,head_pos=0, head_anchor=-1.4, head_length=-0.12,head_width=0.06):
    s=np.array(start)
    e=np.array(end)
    v=e-s 
    l=np.dot(v,v)**0.5
    if l==0:
      return
    ev2=v/l
    vdir=np.argmax(abs(v))
    if v[vdir]==0:
      return
    v_=v.copy()
    v_[vdir]=0
    v0=np.cross(v,v_)
    l0=np.dot(v0,v0)**0.5
    if l0==0.0:
      if vdir!=2:
        ev0=np.array((0.0,0.0,1.0))
      else:
        ev0=np.array((1.0,0.0,0.0))
      l1=1.0
    else:
      ev0=v0/l0
    ev1=np.cross(ev2,ev0)
    #ev2: unit vector in arrow direction
    #ev0: unit vector in arrow head sideways direction
    #ev1: unit vector perpendiculat to arrow head plane
    p_tip=s+v*head_pos+ev2*head_length*head_anchor
    arrow_points=(s+0.23*v,p_tip,p_tip+ev0*0.5*head_width-ev2*head_length,p_tip-ev0*0.5*head_width-ev2*head_length,p_tip,e,)
    return arrow_points
  
def plotCube(tree,azim=-50*pi/180,elev=10*pi/180,camera=50,width=4.5,height=4.5):
  from matplotlib import pyplot as plt
  from matplotlib.collections import LineCollection
  def project(coord3D,M,camera=0):
    coord2d=np.dot(coord3D,M)
    perspective=1 if camera==0 else camera/(camera-coord2d[2])
    return coord2d[:2]*perspective,perspective,coord2d[2]
  plt.close()
  fig = plt.figure()
  fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
  ax = fig.add_subplot(111)
  fig.set_size_inches(width,height)
  scale=height/5
  M=scale*projectionMatrix(-pi/2-azim,elev-pi/2)
  for node in tree.nodes():
    c=node.label
    special_characters={'ż':r'$\dot{\rm{z}}$', 'ĝ':r'$+\hat{\rm{g}}$','ŝ':r'$\hat{\rm{s}}$', 
      'ĵ':r'$\hat{\rm{j}}$', ' ':r'_'}
    c=special_characters.get(c,c)
    coord2d,perspective,zorder=project(node.cubecoord,M,camera=camera)
    for delta in ((0,0,-1,0,0,1,0,0,0,)[-(i+4):-(i+1)] for i in range(6)):
      x,y,z=((node.cubeindex[i]+delta[i]) % 4 for i in range(3))
      neighbournode=tree.cubenodes[x][y][z]
      if neighbournode.right is node:
        style=dict(arrow_direction=1 if node.reverse else -1,linestyle='-',c='b',linewidth=1.5)
      elif neighbournode.left is node:
        style=dict(arrow_direction=1 if node.reverse else -1,linestyle='-',c='r',linewidth=1.5)
      elif neighbournode.parent is node:
        if neighbournode.isrightchild:
          style=dict(arrow_direction=-1 if node.reverse else 1,linestyle='-',c='b',linewidth=1.5)
        else:
          style=dict(arrow_direction=-1 if node.reverse else 1,linestyle='-',c='r',linewidth=1.5)
      elif delta[2]==0:
        style=dict(arrow_direction=0,linestyle='-',c='gray',linewidth=0.5)
      else:
        continue
      arrow_direction=style.pop('arrow_direction')
      p2=[xi+0.5*di for xi,di in zip(node.cubecoord,delta,)]
      if (arrow_direction==1) and (sum(abs(xi)>1.75 for xi in p2)==0):
        arrow_direction=0
      if arrow_direction!=0:
        segments=[project(p3D,M,camera=camera)[0] for p3D in arrow3Dsegments(node.cubecoord,p2,head_pos=(arrow_direction+1)/2, head_anchor=1.4*(arrow_direction-1)/2, head_length=arrow_direction*0.12,head_width=0.06)]
      else:
        segments=[project(p3D,M,camera=camera)[0] for p3D in (node.cubecoord,p2,)]
      ax.plot(*zip(*segments),zorder=zorder,**style) 
    ax.scatter(*coord2d, c='w', marker='o', s=180*perspective, edgecolor= 'grey' if node.parent==None and node.left==None and node.right==None else 'lime' if (node.level&1)==0 else 'magenta',zorder=zorder+0.25)
    ax.text(*coord2d,c, size=10*perspective, zorder=zorder+0.5,  color='k',ha='center',va='center')
  plt.ylim(-height/2,height/2)
  plt.xlim(-width/2,width/2)
  plt.axis('off')
  plt.show()

def makerom1():
  morserom=[f(morse.get(chr(i).lower(),0)) for f in (reversemorse,) for i in range(256)]
  with open('morsecode.bin','wb') as f: 
    f.write(bytes(morserom))
def makerom2():
  morserom=[b if b<256 else 32 for i in range(128) for c in (rmorse.get(i,'\x00'),) for b in(ord(c if c!='' else '\x00'),)]
  with open('rmorse.bin','wb') as f: 
    f.write(bytes(morserom)) 
class ROM:
  def __len__(self):
    return 1<<(9+1)#number of used address lines
  def __iter__(self):
    return(self[i] for i in range(len(self)))
  def __getitem__(self,i):
    state=i&0b01111111
    clock=(i>>7)&1#the morse signal
    data=(i>>8)&1
    reset=(i>>9)&1
    dash=data==1
    dot=data==0
    if reset:
        if state & 64:
          return state^64
        if receivertree.nodeForCubeId[state&0b111111]==None:
          newstate=state^((1<<state.bit_length())>>1) 
          return 128|newstate if (state!=0 or clock or data) else newstate
        node=receivertree.nodeForCubeId[state&0b111111]
        nextnode=node.parent
        if nextnode==None or nextnode.cubeid==None:
          newstate=state^((1<<state.bit_length())>>1) 
          return 128|newstate if (state!=0 or clock or data) else newstate
        nextstate=nextnode.cubeid
        statechange=(state^nextstate)&0b111111
        assert count_bits(statechange)<=1
        newstate=state^statechange
        return newstate|128 if (newstate!=0 or clock or data) else newstate&127
    if clock:
      if parity(state):
        newstate=state^64
        return newstate|128
      else:
        return state|128 
    if not clock:
      if not parity(state):
        if receivertree.nodeForCubeId[state&0b111111]==None:
          return (state|128 if state!=0 else state&127)^64
        node=receivertree.nodeForCubeId[state&0b111111]
        nextnode=node.left if data==0 else node.right
        if nextnode==None or nextnode.cubeid==None:
          return (state|128 if state!=0 else state&127)^64
        nextstate=nextnode.cubeid
        statechange=(state^nextstate)&0b111111
        assert count_bits(statechange)<=1
        if statechange==0:
          statechange=64
        newstate=state^statechange
        return newstate|128 if newstate!=0 else newstate&127
      else:
        return state|128 if state!=0 else state&127
        
hypercubecoord2chr=[ ord(c) if ord(c)<256 else 32 for state in range(64) for c in (receivertree.nodeForCubeId[state].label,)]
nodeid=[receivertree.nodeForCubeId[state].id for state in range(64)]

r=ROM()
assert max([count_bits((x^i)&0x7f) for i,x in enumerate(r)])<=1
state=0
for dd in (1,1,1,0,0,1,0):
  state=r[state&127]
  print(f'{state:08b} {receivertree.nodeForCubeId[state&0b111111].id} {receivertree.nodeForCubeId[state&0b111111].label}')
  state=r[(1<<7)+(state&127)]
  print(f'{state:08b} {receivertree.nodeForCubeId[state&0b111111].id} {receivertree.nodeForCubeId[state&0b111111].label}')
  state=r[(1<<8)*dd+(state&127)]
  print(f'{state:08b} {receivertree.nodeForCubeId[state&0b111111].id} {receivertree.nodeForCubeId[state&0b111111].label}')

state=receivertree.nodeForLabel['!'].cubeid
#state=cubecoord['6']
while state&63!=0:
  state=r[(1<<9)+(state&127)]
  print(f'{state:08b} {receivertree.nodeForCubeId[state&0b111111].id} {receivertree.nodeForCubeId[state&0b111111].label}')

def makerom3():
  morserom=[l[i] for l in (hypercubecoord2chr,nodeid) for i in range(64)]
  with open('hypercube2chr2bin.bin','wb') as f: 
    f.write(bytes(morserom))
def makerom4():
  with open('morse_statemachine.bin','wb') as f: 
    f.write(bytes(ROM()))
    
maxiter=10
l=[maxiter]*1024    
for i in range(len(r)):
  state=i&0x7f
  input=i^state#clear state bits, leave remaining input bits
  for k in range(maxiter):
    oldstate=state
    state=r[input|state]&0x7f
    if state==oldstate: 
      l[i]=k
      break
assert max(l)<maxiter

def grayToInt(gray):
  mask=gray>>1
  result=gray
  while mask!=0:
    result^=mask
    mask>>=1
  return result

def intToGray(i):
  return i^(i>>1)

  
def gray2count(gray,n):
  if n%2==1: raise Exception(f'n must be even')
  bl=n.bit_length( )
  max=1<<bl
  parts=1
  while n&parts==0:
    parts<<=1
  parts>>1
  delta_n=2*n//parts
  offs=delta_n//2
  delta_m=2*max//parts
  i=grayToInt(gray)
  d,r=divmod((i+offs),delta_m)
  if r>=delta_n:
    raise Exception(f'remainder too large: must be less than {delta_n}, but is {r}')
  spr=d*delta_n+r-offs 
  return spr
    
def count2gray(count,n):
  if n%2==1: raise Exception(f'n must be even')
  bl=n.bit_length( )
  max=1<<bl
  parts=1
  while n&parts==0:
    parts<<=1
  parts>>1
  delta_n=2*n//parts
  offs=delta_n//2
  delta_m=2*max//parts
  d,r=divmod((count+offs),delta_n)
  spr=d*delta_m+r-offs
  return intToGray(spr)
  
def serialbit(ascii,counter):
  if counter in (0,11,12,19,23): #19=bit5 for space (end of word)
    return 1
  if (counter >=2) and (counter<=9):
    return 1 if (ascii>>(counter-2))&1 else 0
  if counter==10:
    return 1 if parity(ascii) else 0 #odd parity
  return 0
  
class UART_ROM(ROM):
  def __len__(self):
    return 1<<(6+3+6)#2**(number of used address lines - 1) 6 input data, clk, char, word, state(counter 0..63)
  def __getitem__(self,addr):
    #addr bits:
    # 0.. 5 : step counter as gray code, 48 steps
    #     6 : clk
    #     7 : chr_ready 
    #     8 : word_ready
    # 9..14 : input chr code (one of 64 possible codes)
    #step0: wait for chr_ready -> continue to step1
    #step1..22:count up
    #step23:set reset, wait for ascii==32 (reset complete),
    #wait for not chr_ready and ascii=32-> go to step0
    #or word_ready and ascii==32 ->goto step 24
    #step24..46 : count up
    #step47: wait for not chr_ready-> go to step0
    nsteps=48
    state=addr&0x3f#bit 0..5
    try:
      step=gray2count(state,nsteps) #Exception may be raised if the gray code is not for one of the 48 valid states
    except Exception as e:
      return intToGray((grayToInt(state)+1)%64) #try to fix this by slowly moving to a nearby valid state
    clk=(addr>>6)&1
    char_ready=(addr>>7)&1
    word_ready=(addr>>8)&1
    ascii=hypercubecoord2chr[(addr>>9)&0x3f]
    nextstep=step
    if step==0:
      if char_ready:
        nextstep=1
    if (step>=1) and (step<=22): #transmit the ascii character decoded by the morse state machine
      nextstep=step+1
    if step==23: #this is where the reset for the morse stste machine is set. ascii=32 on reset completion
      if ((not char_ready) and (ascii==32)): #falling edge of chr_ready occured before word_ready -> no space char
        nextstep=0
      elif (word_ready and (ascii==32)): #word_ready means space charecter needs to be transmitter -> continue 
        nextstep=24
    if (step>=24) and (step<=46): #transmit the space character
      nextstep=step+1
    if step==47:
      if not char_ready:#wait for the falling edge to prevent retransmission of the same character repeatedly
        nextstep=0       
    nextstate=count2gray(nextstep,nsteps)
    if parity(nextstate)==clk: #change the state synchronously
      if count_bits(state^nextstate)>1:
        print(locals())
      step=nextstep
      state=nextstate 
    return state|(serialbit(ascii,step//2)<<6)|((1<<7) if (step in (22,23))else 0)
    
r=UART_ROM()
assert max([count_bits((x^i)&0x3f) for i,x in enumerate(r)])<=1
maxiter=10
l=[maxiter]*len(r)  
for i in range(len(r)):
  state=i&0x3f
  input=i^state#clear state bits, leave remaining input bits
  for k in range(maxiter):
    oldstate=state
    state=r[input|state]&0x3f
    if state==oldstate: 
      l[i]=k
      break
assert max(l)<maxiter

  
def makerom5():
  with open('morse_uart.bin','wb') as f: 
    f.write(bytes(UART_ROM()))

class MORSE_ROM(ROM):
  def __len__(self):
    return (1<<9)#1 clk, 7 state bits (6 bits for 64 states + 1 parity bit),256 byte lookup Table
  def __getitem__(self,address):
    if (address>>8)==1:
      state=transmittertree.nodeForLabel.get(chr(address&255).lower(),None).cubeid
      if not parity(state):
        state^=64
      return state
    clk=(address>>7)&1
    state=address & 127
    p=parity(state)
    coord=state & 63
    if transmittertree.nodeForCubeId[coord]==None:# if this coordinate is unused, do nothing:
      return state
    node=transmittertree.nodeForCubeId[coord]
    if p: #even parity
      if coord != 0:
        nextstate=state^64 #make parity odd leave coord unchanged, only if end not yet reached
      else:
        nextstate=state
    else: #odd parity
      nextstate=(state&64) | node.parent.cubeid  
    if clk==p:
      state=nextstate
    p=parity(state)
    coord=state & 63
    node=transmittertree.nodeForCubeId[coord]
    signal=128 if (p or node.isrightchild) and ((node.parent!=None) and (node.parent.id!=0)) else 0
    return signal | state
#with open('MORSE_ROM.bin','wb') as f: f.write(bytes(MORSE_ROM())) 

def Morse(c):
  mr=MORSE_ROM()
  asc=ord(c)
  state=mr[256+asc] if asc<256 else 0
  clk=0
  while True:
    state=mr[clk | (state & 0x7f)]
    state=mr[clk | (state & 0x7f)]
    if state==0:
      break
    yield '–' if state&128 else '␣' 
    clk^=128
  

if __name__=='__main__':
  plotCube(receivertree)
  plotTree(receivertree)
  
  plotCube(transmittertree) 
  plotTree(transmittertree)
  
  for c in 'ab c^de':
    print(c,''.join(Morse(c)))

