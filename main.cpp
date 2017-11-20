#include<iostream>
#include<time.h>
#include<math.h>
#include<stdlib.h>

using namespace std;

double act(double z)
{
  return 1.0/(1.0+exp(-z));
}

double dact(double a)
{
  return a*(1.0-a);
}

double derror(double t,double a)
{
  return (a-t);
}

double mse(double *t,double *a,int n)
{
  double sum=0.0;
  for(int i=0;i<n;i++)
    sum+=0.5*pow(t[i]-a[i],2);
  return sum;
}

int main(void)
{
  cout.precision(8);
  int n_inputs,n_outputs,n_instances,n_layers,n_epoch;
  double alpha,aleatorio,error;
  
  cout<<" Artificial Neural Network with n-layers\n=========================================\n\n";
  
  //topology
  cout<<"Ingrese Nro de Capas";cin>>n_layers;
  int *layer = new int[n_layers];
  for(int i=0;i<n_layers;i++)
  {
    cout<<"\nIngrese el Nro de Neuronas en la capa "<<i<<": ";
    cin>>layer[i];
  }
  
  cout<<"\nIngrese el Nro de Instancias: ";cin>>n_instances;
  
  //create matrix of inputs and outputs
  double **input = new double*[n_instances];
  double **output = new double*[n_instances];
  for(int i=0;i<n_instances;i++)
  {
    input[i] = new double[layer[0]];
    output[i] = new double[layer[n_layers-1]];
  }
  cout<<"\nIngrese la matriz de instancias*(inputs+outputs)\n";
  
  //read matrix of inputs and outputs
  for(int i=0;i<n_instances;i++)
  {
    for(int j=0;j<layer[0];j++)
      cin>>input[i][j];
    for(int j=0;j<layer[n_layers-1];j++)
      cin>>output[i][j];
  }
  
  cout<<"\nIngrese el Nro de epocas\n";cin>>n_epoch;
  
  cout<<"\nIngrese el Factor de Aprendizaje\n";cin>>alpha;
  
  //create structure of data for forward
  double z;
  double **a = new double*[n_layers];
  double **b = new double*[n_layers-1];
  double ***w = new double**[n_layers-1];
  for(int i=1;i<n_layers;i++)
    a[i] = new double[layer[i]];
  for(int i=0;i<n_layers-1;i++)
  {
    b[i] = new double[layer[i+1]];
    for(int j=0;j<layer[i+1];j++)
      b[i][j]=((double)rand()/(RAND_MAX));
  }
  for(int i=0;i<n_layers-1;i++)
  {
    w[i] = new double*[layer[i]];
    for(int j=0;j<layer[i];j++)
    {
      w[i][j] = new double[layer[i+1]];
      for(int k=0;k<layer[i+1];k++)
        w[i][j][k]=((double)rand()/(RAND_MAX));
    }
  }
  
	/*b[0][0]=0.05;
	b[0][1]=0.1;
	b[1][0]=0.15;
	b[1][1]=0.2;
	b[2][0]=0.25;
	//b[2][1]=0.3;
	w[0][0][0]=0.35;
	w[0][0][1]=0.4;
	w[0][1][0]=0.45;
	w[0][1][1]=0.5;
	w[1][0][0]=0.55;
	w[1][0][1]=0.6;
	w[1][1][0]=0.65;
	w[1][1][1]=0.7;
	w[2][0][0]=0.75;
	//w[2][0][1]=0.8;
	w[2][1][0]=0.85;
	//w[2][1][1]=0.9;*/
  
  //create structure of data for backward
  double **d_b = new double*[n_layers-1];
  for(int i=0;i<n_layers-1;i++)
    d_b[i] = new double[layer[i+1]];

  for(int epoch=0;epoch<n_epoch;epoch++)
  {
    error=0.0;
    for(int instance=0;instance<n_instances;instance++)
    {
      //forward propagation

      a[0]=input[instance];
      for(int i=0;i<n_layers-1;i++)
        for(int j=0;j<layer[i+1];j++)
        {
          z=b[i][j];
          for(int k=0;k<layer[i];k++)
            z+=a[i][k]*w[i][k][j];
          a[i+1][j]=act(z);
        }

      //calculate error
      error+=mse(output[instance],a[n_layers-1],layer[n_layers-1]);
      
      //backward propagation
      //for the last layer
      int m=n_layers-2;
      for(int i=0;i<layer[m+1];i++)
      {
        d_b[m][i]=derror(output[instance][i],a[m+1][i])*dact(a[m+1][i]);
      }

      //for another layers
      for(int m=n_layers-3;m>-1;m--)
        for(int i=0;i<layer[m+1];i++)
        {
          z=0.0;
          for(int j=0;j<layer[m+2];j++)
            z+=w[m+1][i][j]*d_b[m+1][j];
          d_b[m][i]=z*dact(a[m+1][i]);
        }
        
      //new weight and bias ///////falta verificar
      for(m=0;m<n_layers-1;m++)
        for(int i=0;i<layer[m];i++)
          for(int j=0;j<layer[m+1];j++)
          {
            w[m][i][j]-=alpha*d_b[m][j]*a[m][i];
            b[m][j]-=alpha*d_b[m][j];
          }
    }
    cout<<endl<<epoch<<" : "<<error/n_instances;
  }

  return 0;
}
