/*
   Scott's AKE Client/Server testbed

   See http://eprint.iacr.org/2002/164

   Compile as 
   cl /O2 /GX /DZZNS=8 ake18kssx.cpp zzn18.cpp zzn6.cpp ecn3.cpp zzn3.cpp big.cpp zzn.cpp ecn.cpp miracl.lib

   KSS k=18 Curve - R-ate pairing

   The KSS curve generated is generated from a 64-bit x parameter
   This version implements the R-ate pairing

   NOTE: Irreducible polynomial is of the form x^18+2

   See kss18.cpp for a program to generate suitable kss18 curves

   Modified to prevent sub-group confinement attack
*/

#include <iostream>
#include <fstream>
#include <string.h>
#include "ecn.h"
#include <ctime>
#include "ecn3.h"
#include "zzn18.h"

using namespace std;
namespace primihub{
    namespace falcon
{
#ifdef MR_COUNT_OPS
extern "C"
{
    int fpc=0;
    int fpa=0;
    int fpx=0;
	int fpm2=0;
	int fpi2=0;
}
#endif

#if MIRACL==64
Miracl precision(8,0); 
#else
Miracl precision(16,0);
#endif

// Non-Residue. Irreducible Poly is binomial x^18-NR

#define NR -2

// Using SHA-256 as basic hash algorithm

#define HASH_LEN 32

//
// Ate Pairing Code
//

// Note - this representation depends on p-1=12 mod 18

void set_frobenius_constant(ZZn &X)
{ // Note X=NR^[(p-13)/18];
    Big p=get_modulus();
	X=pow((ZZn)NR,(p-13)/18);	
}

void endomorph(ECn &A,ZZn &Beta)
{ // apply endomorphism (x,y) = (Beta*x,y) where Beta is cube root of unity
	ZZn x;
	x=(A.get_point())->X;
	x*=Beta;
	copy(getbig(x),(A.get_point())->X);
}

//
// This calculates p.A quickly using Frobenius
// 1. Extract A(x,y) from twisted curve to point on curve over full extension, as X=i^2.x and Y=i^3.y
// where i=NR^(1/k)
// 2. Using Frobenius calculate (X^p,Y^p)
// 3. map back to twisted curve
// Here we simplify things by doing whole calculation on the twisted curve
//
// Note we have to be careful as in detail it depends on w where p=w mod k
// In this case w=13
//

ECn3 psi(ECn3 &A,ZZn &W,int n)
{ 
	int i;
	ECn3 R;
	ZZn3 X,Y;
	ZZn FF;
// Fast multiplication of A by q^n
    A.get(X,Y);

	FF=NR*W*W;
	for (i=0;i<n;i++)
	{ // assumes p=13 mod 18
		X.powq(); X=tx(FF*X);
		Y.powq(); Y*=(ZZn)get_mip()->sru;
	}
    R.set(X,Y);
	return R;
}

//
// Line from A to destination C. Let A=(x,y)
// Line Y-slope.X-c=0, through A, so intercept c=y-slope.x
// Line Y-slope.X-y+slope.x = (Y-y)-slope.(X-x) = 0
// Now evaluate at Q -> return (Qy-y)-slope.(Qx-x)
//

ZZn18 line(ECn3& A,ECn3& C,ZZn3& slope,ZZn& Qx,ZZn& Qy)
{
     ZZn18 w;
     ZZn6 nn,dd;
     ZZn3 X,Y;

     A.get(X,Y);
	
	 nn.set(Qy,Y-slope*X);
	 dd.set(slope*Qx);
	 w.set(nn,dd);

     return w;
}

//
// Add A=A+B  (or A=A+A) 
// Return line function value
//

ZZn18 g(ECn3& A,ECn3& B,ZZn& Qx,ZZn& Qy)
{
    ZZn3 lam;
    ZZn18 r;
    ECn3 P=A;

// Evaluate line from A
    A.add(B,lam,NULL,NULL);
    if (A.iszero())   return (ZZn18)1; 
    r=line(P,A,lam,Qx,Qy);

    return r;
}

ZZn18 Frobenius(const ZZn18& W,ZZn& X,int n)
{
	int i;
	ZZn18 V=W;
	for (i=0;i<n;i++)
		V.powq(X);
	return V;
}

// Automatically generated by Luis Dominquez

ZZn18 HardExpo(ZZn18 &f3x0, ZZn &X, Big &x){
//vector=[ 3, 5, 7, 14, 15, 21, 25, 35, 49, 54, 62, 70, 87, 98, 112, 245, 273, 319, 343, 434, 450, 581, 609, 784, 931, 1407, 1911, 4802, 6517 ]
  ZZn18 xA;
  ZZn18 xB;
  ZZn18 t0;
  ZZn18 t1;
  ZZn18 t2;
  ZZn18 t3;
  ZZn18 t4;
  ZZn18 t5;
  ZZn18 t6;
  ZZn18 t7;
  ZZn18 f3x1;
  ZZn18 f3x2;
  ZZn18 f3x3;
  ZZn18 f3x4;
  ZZn18 f3x5;
  ZZn18 f3x6;
  ZZn18 f3x7;

  f3x1=pow(f3x0,x);
  f3x2=pow(f3x1,x);
  f3x3=pow(f3x2,x);
  f3x4=pow(f3x3,x);
  f3x5=pow(f3x4,x);
  f3x6=pow(f3x5,x);
  f3x7=pow(f3x6,x);

  xA=Frobenius(inverse(f3x1),X,2);
  xB=Frobenius(inverse(f3x0),X,2);
  t0=xA*xB;
  xB=Frobenius(inverse(f3x2),X,2);
  t1=t0*xB;
  t0=t0*t0;
  xB=Frobenius(inverse(f3x0),X,2);
  t0=t0*xB;
  xB=Frobenius(f3x1,X,1);
  t0=t0*xB;
  xA=Frobenius(inverse(f3x5),X,2)*Frobenius(f3x4,X,4)*Frobenius(f3x2,X,5);
  //xB=Frobenius(f3x1,X,1);
  t5=xA*xB;
  t0=t0*t0;
  t3=t0*t1;
  xA=Frobenius(inverse(f3x4),X,2)*Frobenius(f3x1,X,5);
  xB=Frobenius(f3x2,X,1);
  t1=xA*xB;
  xA=xB;//Frobenius(f3x2,X,1);
  xB=xA; //xB=Frobenius(f3x2,X,1);
  t0=xA*xB;
  xB=Frobenius(f3x2,X,4);
  t0=t0*xB;
  xB=Frobenius(f3x1,X,4);
  t2=t3*xB;
  xB=Frobenius(inverse(f3x1),X,2);
  t4=t3*xB;
  t2=t2*t2;
  xB=Frobenius(inverse(f3x2),X,3);
  t3=t0*xB;
  xB=inverse(f3x2);
  t0=t3*xB;
  t4=t3*t4;
  xB=Frobenius(inverse(f3x3),X,3);
  t0=t0*xB;
  t3=t0*t2;
  xB=Frobenius(inverse(f3x3),X,2)*Frobenius(f3x0,X,5);
  t2=t3*xB;
  t3=t3*t5;
  t5=t3*t2;
  xB=inverse(f3x3);
  t2=t2*xB;
  xA=Frobenius(inverse(f3x6),X,3);
  //xB=inverse(f3x3);
  t3=xA*xB;
  t2=t2*t2;
  t4=t2*t4;
  xB=Frobenius(f3x3,X,1);
  t2=t1*xB;
  xA=xB; //xA=Frobenius(f3x3,X,1);
  xB=Frobenius(inverse(f3x2),X,3);
  t1=xA*xB;
  t6=t2*t4;
  xB=Frobenius(f3x4,X,1);
  t4=t2*xB;
  xB=Frobenius(f3x3,X,4);
  t2=t6*xB;
  xB=Frobenius(inverse(f3x5),X,3)*Frobenius(f3x5,X,4);
  t7=t6*xB;
  t4=t2*t4;
  xB=Frobenius(f3x6,X,1);
  t2=t2*xB;
  t4=t4*t4;
  t4=t4*t5;
  xA=inverse(f3x4);
  xB=Frobenius(inverse(f3x4),X,3);
  t5=xA*xB;
//  xB=Frobenius(inverse(f3x4),X,3);
  t3=t3*xB;
  xA=Frobenius(f3x5,X,1);
  xB=xA; //xB=Frobenius(f3x5,X,1);
  t6=xA*xB;
  t7=t6*t7;
  xB=Frobenius(f3x0,X,3);
  t6=t5*xB;
  t4=t6*t4;
  xB=Frobenius(inverse(f3x7),X,3);
  t6=t6*xB;
  t0=t4*t0;
  xB=Frobenius(f3x6,X,4);
  t4=t4*xB;
  t0=t0*t0;
  xB=inverse(f3x5);
  t0=t0*xB;
  t1=t7*t1;
  t4=t4*t7;
  t1=t1*t1;
  t2=t1*t2;
  t1=t0*t3;
  xB=Frobenius(inverse(f3x3),X,3);
  t0=t1*xB;
  t1=t1*t6;
  t0=t0*t0;
  t0=t0*t5;
  xB=inverse(f3x6);
  t2=t2*xB;
  t2=t2*t2;
  t2=t2*t4;
  t0=t0*t0;
  t0=t0*t3;
  t1=t2*t1;
  t0=t1*t0;
//  xB=inverse(f3x6);
  t1=t1*xB;
  t0=t0*t0;
  t0=t0*t2;
  xB=f3x0*inverse(f3x7);
  t0=t0*xB;
//  xB=f3x0*inverse(f3x7);
  t1=t1*xB;
  t0=t0*t0;
  t0=t0*t1;

  return t0;
}

//
// R-ate Pairing - note denominator elimination has been applied
//
// P is a point of order q. Q(x,y) is a point of order q. 
// Note that P is a point on the sextic twist of the curve over Fp^2, Q(x,y) is a point on the 
// curve over the base field Fp
//

BOOL fast_pairing(ECn3& P,ZZn& Qx,ZZn& Qy,Big &x,ZZn &X,ZZn18& r)
{ 
    ECn3 A,m2A,dA;
    int i,nb;
    Big d;
    ZZn18 rd;

#ifdef MR_COUNT_OPS
fpc=fpa=fpx=0;
#endif

	A=P;         // remember A

	d=(x/7);  

	nb=bits(d);
	r=1;
	r.mark_as_miller();
	for (i=nb-2;i>=0;i--)
    {
        r*=r;  
        r*=g(A,A,Qx,Qy);
        if (bit(d,i)) 
            r*=g(A,P,Qx,Qy);  
    }
	rd=r;
	dA=A;

	r*=r;
	r*=g(A,A,Qx,Qy);

	m2A=A;

	rd*=r;
	rd*=g(A,dA,Qx,Qy);

	r*=Frobenius(rd,X,6);

	A=psi(A,X,6);
	r*=g(A,m2A,Qx,Qy);
#ifdef MR_COUNT_OPS
cout << "Miller fpc= " << fpc << endl;
cout << "Miller fpa= " << fpa << endl;
cout << "Miller fpx= " << fpx << endl;
fpa=fpc=fpx=0;
#endif

// final exponentiation
	rd=r;
    r.conj();
    r/=rd;    // r^(p^9-1)
	r.mark_as_regular(); // no longer "miller"
	rd=r;
	r.powq(X); r.powq(X); r.powq(X); r*=rd; //r^(p^3+1)

	r.mark_as_unitary();
	r=HardExpo(r,X,x);

#ifdef MR_COUNT_OPS
cout << "FE fpc= " << fpc << endl;
cout << "FE fpa= " << fpa << endl;
cout << "FE fpx= " << fpx << endl;
fpa=fpc=fpx=0;
#endif

    return TRUE;
}

//
// ecap(.) function
//

BOOL ecap(ECn3& P,ECn& Q,Big& x,ZZn &X,ZZn18& r)
{
    BOOL Ok;
    Big xx,yy;
    ZZn Qx,Qy;

    Q.get(xx,yy); Qx=xx; Qy=yy;

    Ok=fast_pairing(P,Qx,Qy,x,X,r);

    if (Ok) return TRUE;
    return FALSE;
}

//
// Hash functions
// 

Big H2(ZZn18 x)
{ // Compress and hash an Fp18 to a big number
    sha256 sh;
    ZZn6 u;
    ZZn3 h,l;
    Big a,hash,p;
	ZZn xx[6];
    char s[HASH_LEN];
    int i,j,m;

    shs256_init(&sh);
    x.get(u);  // compress to single ZZn6
    u.get(l,h);
	l.get(xx[0],xx[1],xx[2]);
	h.get(xx[3],xx[4],xx[5]);
    
    for (i=0;i<6;i++)
    {
        a=(Big)xx[i];
        while (a>0)
        {
            m=a%256;
            shs256_process(&sh,m);
            a/=256;
        }
    }
    shs256_hash(&sh,s);
    hash=from_binary(HASH_LEN,s);
    return hash;
}

Big H1(char *string)
{ // Hash a zero-terminated string to a number < modulus
    Big h,p;
    char s[HASH_LEN];
    int i,j; 
    sha256 sh;

    shs256_init(&sh);

    for (i=0;;i++)
    {
        if (string[i]==0) break;
        shs256_process(&sh,string[i]);
    }
    shs256_hash(&sh,s);
    p=get_modulus();
    h=1; j=0; i=1;
    forever
    {
        h*=256; 
        if (j==HASH_LEN)  {h+=i++; j=0;}
        else         h+=s[j++];
        if (h>=p) break;
    }
    h%=p;
    return h;
}

// Hash and map a Server Identity to a curve point E_(Fp3)

ECn3 hash_and_map3(char *ID)
{
    int i;
    ECn3 S;
    ZZn3 X;
 
    Big x0=H1(ID);
    forever
    {
        x0+=1;
        X.set((ZZn)0,(ZZn)x0,(ZZn)0);
        if (!S.set(X)) continue;
        break;
    }
  
    return S;
}     

// Hash and Map a Client Identity to a curve point E_(Fp) of order q

ECn hash_and_map(char *ID,Big cf)
{
    ECn Q;
    Big x0=H1(ID);
    while (!Q.set(x0,x0)) x0+=1;
    Q*=cf;
    return Q;
}

// Faster Hashing to G2 - Fuentes-Castaneda, Knapp and Rodriguez-Henriquez

ECn3 HashG2(ECn3& Qx0,Big &x,ZZn&F)
{
	ECn3 Qx0_;
	ECn3 Qx1;
	ECn3 Qx1_;
	ECn3 Qx2;
	ECn3 Qx2_;
	ECn3 Qx3;
	ECn3 t1;
	ECn3 t2;
	ECn3 t3;
	ECn3 t4;
	ECn3 t5;
	ECn3 t6;

	Qx0_=-Qx0;
	Qx1=x*Qx0;
	Qx1_=-Qx1;
	Qx2=x*Qx1;
	Qx2_=-Qx2;
	Qx3=x*Qx2;

	t1=Qx0;
	t2=psi(Qx1_,F,2);
	t3=Qx1+psi(Qx1,F,5);
	t4=psi(Qx1,F,3)+psi(Qx2,F,1)+psi(Qx2_,F,2);
	t5=psi(Qx0_,F,4);
	t6=psi(Qx0,F,1)+psi(Qx0,F,3)+psi(Qx2_,F,4)+psi(Qx2,F,5)+psi(Qx3,F,1);

	t2+=t1;  // Olivos addition sequence
	t1+=t1;
	t1+=t3;
	t1+=t2;
	t4+=t2;
	t5+=t1;
	t4+=t1;
	t5+=t4;
	t4+=t6;
	t5+=t5;
	t5+=t4;

	return t5;
}

// Use Galbraith & Scott Homomorphism idea ...

void galscott(Big &e,Big &r,Big WB[6],Big B[6][6],Big u[6])
{
	int i,j;
	Big v[6],w;

	for (i=0;i<6;i++)
	{
		v[i]=mad(WB[i],e,(Big)0,r,w);
		u[i]=0;
	}

	u[0]=e;
	for (i=0;i<6;i++)
	{
		for (j=0;j<6;j++)
			u[i]-=v[j]*B[j][i];
	}
	return;
}

// GLV method

void glv(Big &e,Big &r,Big W[2],Big B[2][2],Big u[2])
{
	int i,j;
	Big v[2],w;
	for (i=0;i<2;i++)
	{
		v[i]=mad(W[i],e,(Big)0,r,w);
		u[i]=0;
	}
	u[0]=e;
	for (i=0;i<2;i++)
		for (j=0;j<2;j++)
			u[i]-=v[j]*B[j][i];
	return;
}

// Use GLV endomorphism idea for multiplication in G1

ECn G1_mult(ECn &P,Big &e,ZZn &Beta,Big &r,Big W[2],Big B[2][2])
{
//	return e*P;
	int i;
	ECn Q;
	Big u[2];

	glv(e,r,W,B,u);

	Q=P;
	endomorph(Q,Beta);

	Q=mul(u[0],P,u[1],Q);
	
	return Q;
}

//.. for multiplication in G2

ECn3 G2_mult(ECn3 &P,Big &e,ZZn &X,Big &r,Big WB[6],Big B[6][6])
{
//	return e*P;
	int i;
	ECn3 Q[6];
	Big u[6];
	galscott(e,r,WB,B,u);

	Q[0]=P;
	for (i=1;i<6;i++)
		Q[i]=psi(Q[i-1],X,1);

// deal with -ve multipliers
	for (i=0;i<6;i++)
	{
		if (u[i]<0)
			{u[i]=-u[i];Q[i]=-Q[i];}
	}

// simple multi-addition

	return mul(6,Q,u);
}

//.. and for exponentiation in GT

ZZn18 GT_pow(ZZn18 &res,Big &e,ZZn &X,Big &r,Big WB[6],Big B[6][6])
{
//	return pow(res,e);
	int i,j;
	ZZn18 Y[6];
	Big u[6];

	galscott(e,r,WB,B,u);

	Y[0]=res;
	for (i=1;i<6;i++)
		{Y[i]=Y[i-1]; Y[i].powq(X);}

// deal with -ve exponents
	for (i=0;i<6;i++)
	{
		if (u[i]<0)
			{u[i]=-u[i];Y[i].conj();}
	}

// simple multi-exponentiation
	return pow(6,Y,u);
}

int main()
{
    miracl* mip=&precision;
	ZZn X;
	ZZn3 XX,YY;
    ECn Alice,Bob,sA,sB;
    ECn3 Server,sS;
    ZZn18 sp,ap,bp,res,XXX,YYY;
    Big a,b,s,ss,p,q,x,y,B,cf,t,n,sru,BB[6][6],WB[6],SB[2][2],W[2];
    int i,A;
    time_t seed;

    mip->IOBASE=16;

	x=   (char *)"15000000007004210";  // found by KSS18.CPP - Hamming weight of 9
	t=(pow(x,4) + 16*x + 7)/7;
	q=(pow(x,6) + 37*pow(x,3) + 343)/343;
		
    cf=(49*x*x+245*x+343)/3;
	n=cf*q;
	p=cf*q+t-1; 

//  cout << "p= " << p << endl;
//	cout << "bits(p)= " << bits(p) << endl;
//	cout << "bits(q)= " << bits(q) << endl;

//	p=(pow(x,8) + 5*pow(x,7) + 7*pow(x,6) + 37*pow(x,5) + 188*pow(x,4) + 259*pow(x,3) + 343*pow(x,2) + 1763*x + 2401)/21;

    time(&seed);
    irand((long)seed);

    ecurve((Big)0,(Big)2,p,MR_AFFINE);

//	Big Lambda=pow(x,3)+18;  // cube root of unity mod q
//  Desperately avoiding overflow... - cube root of unity mod p
	Big BBeta=(/*2*pow(x,8)+*/3*pow(x,7)-7*pow(x,6)+46*pow(x,5)+68*pow(x,4)-308*pow(x,3)+189*x*x+145*x-3192)/56;
	BBeta+=x*(pow(x,7)/28);
	BBeta/=3;

	sru=p-BBeta;  // sixth root of unity = -Beta	
	set_zzn3(NR,sru);
	ZZn Beta=BBeta;

// Use standard Gallant-Lambert-Vanstone endomorphism method for G1
	
	W[0]=(x*x*x)/343;        // This is first column of inverse of SB (without division by determinant) 
	W[1]=(18*x*x*x+343)/343;
	
	SB[0][0]=(x*x*x)/343;
	SB[0][1]=-(18*x*x*x+343)/343;
	SB[1][0]=(19*x*x*x+343)/343;
	SB[1][1]=(x*x*x)/343;

// Use Galbraith & Scott Homomorphism idea for G2 & GT ... (http://eprint.iacr.org/2008/117.pdf)

	WB[0]=5*pow(x,3)/49+2;   // This is first column of inverse of BB (without division by determinant) 
	WB[1]=-(x*x)/49;
	WB[2]=pow(x,4)/49+3*x/7;
	WB[3]=-(17*pow(x,3)/343+1);
	WB[4]=-(pow(x,5)/343+2*(x*x)/49);
	WB[5]=5*pow(x,4)/343+2*x/7;

	BB[0][0]=1;      BB[0][1]=0;     BB[0][2]=5*x/7; BB[0][3]=1;   BB[0][4]=0;   BB[0][5]=-x/7; 
	BB[1][0]=-5*x/7; BB[1][1]=-2;    BB[1][2]=0;     BB[1][3]=x/7; BB[1][4]=1;   BB[1][5]=0; 
	BB[2][0]=0;      BB[2][1]=2*x/7; BB[2][2]=1;     BB[2][3]=0;   BB[2][4]=x/7; BB[2][5]=0; 
	BB[3][0]=1;      BB[3][1]=0;     BB[3][2]=x;     BB[3][3]=2;   BB[3][4]=0;   BB[3][5]=0; 
	BB[4][0]=-x;     BB[4][1]=-3;    BB[4][2]=0;     BB[4][3]=0;   BB[4][4]=1;   BB[4][5]=0; 
	BB[5][0]=0;      BB[5][1]=-x;    BB[5][2]=-3;    BB[5][3]=0;   BB[5][4]=0;   BB[5][5]=1;

	cout << "Initialised... " << endl;

    mip->IOBASE=16;

    mip->TWIST=MR_SEXTIC_D;   // map Server to point on twisted curve E(Fp3)
	// See ftp://ftp.computing.dcu.ie/pub/resources/crypto/twists.pdf
	// NOTE: This program only supports D-type twists
	// An M-type twist requires a different "untwisting" operation - see paper above

	set_frobenius_constant(X);

    cout << "Mapping Alice & Bob ID's to points" << endl;
    Alice=hash_and_map((char *)"Alice",cf);
    Bob=  hash_and_map((char *)"Robert",cf); 
    cout << "Mapping Server ID to point" << endl;
    Server=hash_and_map3((char *)"Server");
	Server=HashG2(Server,x,X); // fast multiplication by co-factor

    ss=rand(q);    // TA's super-secret 
	
	cout << "Alice, Bob and the Server visit Trusted Authority" << endl; 

	sS=G2_mult(Server,ss,X,q,WB,BB);
    sA=G1_mult(Alice,ss,Beta,q,W,SB); 
    sB=G1_mult(Bob,ss,Beta,q,W,SB);

    cout << "Alice and Server Key Exchange" << endl;

    a=rand(q);   // Alice's random number
    s=rand(q);   // Server's random number

    // for (i=0;i<1000;i++)
    if (!ecap(Server,sA,x,X,res)) cout << "Trouble" << endl;
    if (pow(res,q)!=(ZZn18)1)
    {
        cout << "Wrong group order - aborting" << endl;
        exit(0);
    }
	ap=GT_pow(res,a,X,q,WB,BB);  

    if (!ecap(sS,Alice,x,X,res)) cout << "Trouble" << endl;
    if (pow(res,q)!=(ZZn18)1)
    {
        cout << "Wrong group order - aborting" << endl;
        exit(0);
    }

	sp=GT_pow(res,s,X,q,WB,BB);

    cout << "Alice  Key= " << H2(GT_pow(sp,a,X,q,WB,BB)) << endl;
    cout << "Server Key= " << H2(GT_pow(ap,s,X,q,WB,BB)) << endl;

    cout << "Bob and Server Key Exchange" << endl;

    b=rand(q);   // Bob's random number
    s=rand(q);   // Server's random number

    if (!ecap(Server,sB,x,X,res)) cout << "Trouble" << endl;
    if (pow(res,q)!=(ZZn18)1)
    {
        cout << "Wrong group order - aborting" << endl;
        exit(0);
    }
 
	bp=GT_pow(res,b,X,q,WB,BB);

    if (!ecap(sS,Bob,x,X,res)) cout << "Trouble" << endl;
    if (pow(res,q)!=(ZZn18)1)
    {
        cout << "Wrong group order - aborting" << endl;
        exit(0);
    }

	sp=GT_pow(res,s,X,q,WB,BB);

    cout << "Bob's  Key= " << H2(GT_pow(sp,b,X,q,WB,BB)) << endl;
    cout << "Server Key= " << H2(GT_pow(bp,s,X,q,WB,BB)) << endl;

    return 0;
}
}
}//primihub