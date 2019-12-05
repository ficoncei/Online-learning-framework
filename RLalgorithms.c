/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   main.c
 * Author: FC251122
 *
 * Created on December 10, 2018, 4:15 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/*ANN*/
double *inputLayerNeurons;
double *outputLayerNeurons;
double *hiddenLayerOneNeurons;
double *hiddenLayerTwoNeurons;
double **ihWeights;
double **hoWeights;
double *ihBiases;
double *hoBiases;
double *ihSums;
double *ihOutputs;
double *hoSums;
double *errorOutputNeuron;
double *expectedOutputNeuron;
double *errorHiddenNeuron;
double **deltaWeight;
double **deltaHiddenSum;

/*ANN*/
int counter=0;
int* policy;

static double **Qvalues;
static double **QvaluesOne;
static double **QvaluesTwo;
int** epsilonmapping;
int** previousmapping;
int** VisitsCount;
double **ActionStateProb;
int* stateVector;
int* actionVector;
double* rewardVector;
double **parameterVector; //theta - policy parameter vector
double **EstimatedQvalues;
double *derivativeQ;
double *derivativeProb;
double *ANNWeightsold;
double *ANNWeightsnew;
double *probWeightsold;
double *probWeightsnew;
double *preferences;
double *preferencesmapping;
double **EstimatedProbabilities;


/*Simulations*/
int **hnruntime;
int **dnruntime;
static double *hnruntimesimulator;
static double *dnruntimesimulator;
static double **batteryevo;
static int **seclevelevo;
static double **discardedpackets;
static double **rewardsevo;
static double *k1;
static double *k2;
static double *k3;
static double *k4;
static double *discardedpacketsovertime;
static double *batteryevoovertime;
static double *rewardevoovertime;
static double *rewardsevosimulation;
static double *discpacketsevosimulation;
static double *rewardevoovertimesimulation;
static double *avgrewardevoovertimesimulation;
double avgdiscardedpackets=0;
double avgrewardsimulation=0;
double avgdiscpacketssimulation=0;
static int eprime[80][4];

/*--------Algorithm control---------*/
int QLEARNING=0;
int DOUBLEQLEARNING=1;
int SARSA=0;
int EXPECTEDSARSA=0;
int NSTEPSARSA=0;
int ACTORCRITIC=0;
/*-----Value 1 to select an algo----*/

int NAIVEPOLICY=0;
int FIXEDSECLEVEL=4;
int ENERGYVARIATION=1;

/*-------Simulation control---------*/
int SIMULATION=0;
/*----------1>Simulation ON---------*/

/*------------ANN control-----------*/
int ANNCONTROL=0;
/*------------ANN control-----------*/

int maxAllowedActions=4;
int upper=0,lower=0;
static int Actionset[]={-3,-2,-1,0,1,2,3};  /*1:7 for matrix subscript index*/
static int lengthActionset=sizeof(Actionset)/4;
/*-------Instantiate a random number generator-------*/
const gsl_rng_type * T;
gsl_rng * r;
/*gsl_rng_free (r);*/

/*Indexes*/
int episodes=100,epochs=100;
int runtimes=5000;
double epsilon=0.1;
int minenergy=53, maxenergy=53;
int muhn=53, mudn=40;
static double  sigmadn=2, sigmahn=4;
static int levels[] = {1,2,3,4};
double securityf[4];
int packetsizemax=80;
int cnmax=4;

/*ANN*/
int numInputNeurons=2;
int numHiddenNeurons=5;
int numOutputNeurons=7;
int numHiddenNeuronsLayerOne=4;
int numHiddenNeuronsLayerTwo=4;
int numLayers=1;
/*ANN*/


void PrintSarsaPolicyFile (int nStates, int statesdivision);                                                                                                    //SARSA
void PrintQlearningPolicyFile (int nStates, int statesdivision);                                                                                                //Qlearning
void PrintExpectedSarsaPolicyFile (int nStates, int statesdivision);                                                                                            //ExpectedSARSA
void PrintDoubleQlearningPolicyFile (int nStates, int statesdivision);                                                                                          //DOUBLEQLEARNING
void PrintnStepSARSAPolicyFile (int nStates, int statesdivision);                                                                                               //nStepSARSA
int InitialiseQValuesRandomly(int nStates, int statesdivision);                                                                                                 //GENERIC
int deriveEpsilonGreedyPolicyfromQvalues(int nStates, int statesdivision);                                                                                      //GENERIC
int ExtractDeterministicPolicyfromQvalues();                                                                                                                    //SARSA,Qlearning and ExpectedSARSA
int ExtractDeterministicPolicyfromQOneTwovalues(int nStates);                                                                                                   //DOUBLEQLEARNING
int updateQValuesSARSA(int curState, int curAction, int nState, int nAction, double aSarsa, double gSarsa, double R, int statesdivision);                       //SARSA
int updateQValuesQlearning(int curState, int curAction, int nState, double aQlearning, double gQlearning, double R);                                            //Qlearning
int updateQValuesExpectedSARSA(int curState, int curAction, int nState, double aExpectedSARSA, double gExpectedSARSA, double R);                                //ExpectedSARSA
int updateQValuesDoubleQlearning(int curState, int curAction, int nState, double aDoubleQlearning, double gDoubleQlearning, double R);                          //DOUBLEQLEARNING
int updateQValuesnStepSARSA(int epoch, int nstep, int TnStepSARSA, double gammanStepSARSA, double alphanStepSARSA, int maxStates, int blockofstatesperlevel);   //nStepSARSA
int deriveEpsilonGreedyPolicyfromQOneTwovalues(int nStates, int statesdivision);                                                                                //DOUBLEQLEARNING
int simulatePolicy(int batterymax, double alpha, double beta);                                                                                                                             //GENERIC
int selectRandomAction(int state);                                                                                                                              //GENERIC
int selectRandomActionACTORCRITIC(int state, int statesdivision);                                                                                               //ACTORCRITIC
int updateParametersACTORCRITIC(int curState,int nState,int curaction, int naction, int cn, double R, int bmax, double alpha, double beta, int statesdivision,double gACTORCRITIC,int numWeights,double alphaW, double alphaT);    //ACTORCRITIC
double estimateActionValue(int state, int action, int cn, double alpha, double beta, int statesdivision, int bmax);                                             //ACTORCRITIC
double estimateActionProbability(int state, int action, int cn, double alpha, double beta, int statesdivision, int bmax, int lower, int upper);
int InitialiseEstimatedQValuesfromRandomWeights(int nStates, int statesdivision, int bmax, double alpha, double beta);
double* mapWeights(double *weightsvector);
double* softMaxProbabilities(double *logits, int inputsize, int curstate, int statesdivision, int lower, int upper);
int updateDerivativeQ(int curState,int curaction, int numWeights,double actionvalueCurrentState);
int updateDerivativeProb(int curState,int curaction, int numWeights, double probActionCurrentState);
int updateWeightsActionProbabilityFunction(int numWeights, double delta, double alphaTheta);
int updateWeightsActionValueFunction(int numWeights, double delta, double alphaWeights);
int ExtractStochasticPolicyfromPreferences(int nStates,int statesdivision, int bmax, double alpha, double beta);
int PrintACTORCRITICPolicyFile(int maxStates,int statesdivision, int bmax, double alpha, double beta);
void PrintSecLevelVSTimeDataFile();
void PrintDiscPacketsVSTimeDataFile();
void PrintBatteryEvoVSTimeDataFile();
void PrintRewardEvoVSTimeDataFile(double alpha, double beta);
void PrintAvgRewardSimulationVSMuHn(double avgrewardsimulation);
void readHarvesterAndPacketdata();
int getlower(int i, int statesdivision);
int getupper(int i, int statesdivision);
int getcn(int i, int statesdivision);
int sampleAction(int state);

/*ANN*/
double ReLuFunction(double input);
void computeIHWeightedSum();
void computeIHActivation();
void computeHOWeightedSum();
void computeHOActivation();
void addBiasesIH();
void addBiasesHO();
void ForwardPropagate();
double ReLuFunctionDerivative(double input);
double sigmoidDerivative(double input);
void computeErrorOutput();
void computedeltaWeightHO();
void computeErrorHidden();
void adjustWeightsIH();
void adjustWeightsHO();
void BackwardPropagate();
void TrainNetwork(int numItNNTraining);
/*ANN*/

int main(int argc, char** argv) {
int cn=4;
int bmax=384;
int packetsize=40;
int bn,e;
int iState;
//double integ,fract;
static int lengthcn=sizeof(levels)/4;
double alpha=0.8;
double beta=0.3;

//static double alphasteps=10;
//static double betasteps=10;
//static double muhnsteps=10;
//static double mudnsteps=10;
int maxStates;
int currentState=0,nextState=0,currentAction=-1,nextcn=0,nextaction=0;
double alphaSarsa=0.1;
double gammaSarsa=0.9;
double alphaQlearning=0.1;
double gammaQlearning=0.9;
double alphaExpectedSARSA=0.1;
double gammaExpectedSARSA=0.9;
double alphaDoubleQlearning=0.1;
double gammaDoubleQlearning=0.9;
double alphanStepSARSA=0.1;
double gammanStepSARSA=0.9;
int TnStepSARSA=200;
int nstep=2;
double gammaACTORCRITIC=0.9;
double I;
double Reward=0;
double batteryratio;
int overhead;
maxStates=(bmax+1)*lengthcn;
int numWeigths=(numInputNeurons*numHiddenNeurons+numHiddenNeurons*numOutputNeurons);//assumes 1 hidden layer and no biases
int blockofstatesperlevel=ceil((maxStates+1)/lengthcn);

// create random number generator
r = gsl_rng_alloc (gsl_rng_mt19937);

policy = (int*)malloc(maxStates * sizeof(int));

for (int i = 0; i < maxStates; i++){
    policy[i]=0;
}

hnruntime = malloc((maxenergy-minenergy+1) * sizeof(int *));
	for(int i = 0; i < (maxenergy-minenergy+1); i++)
		hnruntime[i] = malloc(epochs * sizeof(int));

	for (int i = 0; i < (maxenergy-minenergy+1); i++)
		for (int j = 0; j < epochs; j++)
                hnruntime[i][j] = 0;

dnruntime = malloc((maxenergy-minenergy+1) * sizeof(int *));
	for(int i = 0; i < (maxenergy-minenergy+1); i++)
		dnruntime[i] = malloc(epochs * sizeof(int));

	for (int i = 0; i < (maxenergy-minenergy+1); i++)
		for (int j = 0; j < epochs; j++)
                dnruntime[i][j] = 0;

hnruntimesimulator = malloc(epochs * sizeof(double));
dnruntimesimulator = malloc(epochs * sizeof(double));

k1 = malloc(epochs * sizeof(double));
k2 = malloc(epochs * sizeof(double));
k3 = malloc(epochs * sizeof(double));
k4 = malloc(epochs * sizeof(double));
discardedpacketsovertime = malloc(epochs * sizeof(double));
batteryevoovertime = malloc(epochs * sizeof(double));
rewardevoovertime = malloc(epochs * sizeof(double));
rewardsevosimulation = malloc(epochs * sizeof(double));
discpacketsevosimulation = malloc(epochs * sizeof(double));
rewardevoovertimesimulation = malloc(epochs * sizeof(double));
avgrewardevoovertimesimulation = malloc(epochs * sizeof(double));

for (int i=0;i<epochs;i++){
    k1[i]=0;
    k2[i]=0;
    k3[i]=0;
    k4[i]=0;
    discardedpacketsovertime[i]=0;
    batteryevoovertime[i]=0;
    rewardevoovertime[i]=0;
}


stateVector=malloc(TnStepSARSA * sizeof(int));
actionVector=malloc(TnStepSARSA * sizeof(int));
rewardVector= malloc(TnStepSARSA * sizeof(double));

for (int i=0;i<TnStepSARSA;i++){
    stateVector[i]=-1;
    actionVector[i]=-1;
    rewardVector[i]=0;
}

derivativeQ = malloc(numWeigths * sizeof(double));
    for(int i = 0; i < numWeigths; i++)
		derivativeQ[i] = 0;

derivativeProb = malloc(numWeigths * sizeof(double));
    for(int i = 0; i < numWeigths; i++)
		derivativeProb[i] = 0;

/*ANN*/
inputLayerNeurons = malloc(numInputNeurons * sizeof(double));
hiddenLayerOneNeurons = malloc(numHiddenNeuronsLayerOne * sizeof(double));
hiddenLayerTwoNeurons = malloc(numHiddenNeuronsLayerTwo * sizeof(double));
outputLayerNeurons = malloc(numOutputNeurons * sizeof(double));
ihBiases = malloc(numHiddenNeurons * sizeof(double));//just in case
hoBiases = malloc(numOutputNeurons * sizeof(double));//just in case
ihSums = malloc(numHiddenNeurons * sizeof(double));
ihOutputs = malloc(numHiddenNeurons * sizeof(double));
hoSums = malloc(numOutputNeurons * sizeof(double));
errorOutputNeuron = malloc(numOutputNeurons * sizeof(double));
expectedOutputNeuron = malloc(numOutputNeurons * sizeof(double));
errorHiddenNeuron = malloc(numHiddenNeurons * sizeof(double));

ANNWeightsold = malloc(numWeigths * sizeof(double));
	for(int i = 0; i < numWeigths; i++)
		ANNWeightsold[i] = gsl_rng_uniform(r);

ANNWeightsnew = malloc(numWeigths * sizeof(double));
	for(int i = 0; i < numWeigths; i++)
		ANNWeightsnew[i] = 0.5;

probWeightsold = malloc(numWeigths * sizeof(double));
	for(int i = 0; i < numWeigths; i++)
		probWeightsold[i] = gsl_rng_uniform(r);

probWeightsnew = malloc(numWeigths * sizeof(double));
	for(int i = 0; i < numWeigths; i++)
		probWeightsnew[i] = 0.5;

preferences = malloc(lengthActionset * sizeof(double));
	for(int i = 0; i < lengthActionset; i++)
		preferences[i] = 0;

preferencesmapping = malloc(maxAllowedActions * sizeof(double));
	for(int i = 0; i < maxAllowedActions; i++)
		preferencesmapping[i] = 0;

deltaWeight = malloc(numHiddenNeurons * sizeof(double *));
	for(int i = 0; i < numHiddenNeurons; i++)
		deltaWeight[i] = malloc(numLayers * sizeof(double));

	for (int i = 0; i < numHiddenNeurons; i++)
		for (int j = 0; j < numLayers; j++)
            deltaWeight[i][j] = 0;

deltaHiddenSum = malloc(numHiddenNeurons * sizeof(double *));
	for(int i = 0; i < numHiddenNeurons; i++)
		deltaHiddenSum[i] = malloc(numLayers * sizeof(double));

	for (int i = 0; i < numHiddenNeurons; i++)
		for (int j = 0; j < numLayers; j++)
                deltaHiddenSum[i][j] = 0;

ihWeights = malloc(numInputNeurons * sizeof(double *));
	for(int i = 0; i < numInputNeurons; i++)
		ihWeights[i] = malloc(numHiddenNeurons * sizeof(double));

	for (int i = 0; i < numInputNeurons; i++)
		for (int j = 0; j < numHiddenNeurons; j++)
                ihWeights[i][j] = gsl_rng_uniform(r);

hoWeights = malloc(numHiddenNeurons * sizeof(double *));
	for(int i = 0; i < numHiddenNeurons; i++)
		hoWeights[i] = malloc(numOutputNeurons * sizeof(double));

	for (int i = 0; i < numHiddenNeurons; i++)
		for (int j = 0; j < numOutputNeurons; j++)
                hoWeights[i][j] = gsl_rng_uniform(r);

    for (int i = 0; i < numHiddenNeurons; i++){
        //hiddenLayerOneNeurons[i]=0;
        ihBiases[i]=0;
        ihSums[i]=0;
        ihOutputs[i]=0;
        errorHiddenNeuron[i]=0;
    }

    for (int i = 0; i < numOutputNeurons; i++){
        hoBiases[i]=0;
        hoSums[i]=0;
    }
/*ANN*/


Qvalues = malloc(maxStates * sizeof(double *));
	for(int i = 0; i < maxStates; i++)
		Qvalues[i] = malloc(lengthActionset * sizeof(double));

	for (int i = 0; i < maxStates; i++)
		for (int j = 0; j < lengthActionset; j++)
                Qvalues[i][j] = -1;

EstimatedQvalues = malloc(maxStates * sizeof(double *));
	for(int i = 0; i < maxStates; i++)
		EstimatedQvalues[i] = malloc(lengthActionset * sizeof(double));

	for (int i = 0; i < maxStates; i++)
		for (int j = 0; j < lengthActionset; j++)
                EstimatedQvalues[i][j] = -1;

EstimatedProbabilities = malloc(maxStates * sizeof(double *));
	for(int i = 0; i < maxStates; i++)
		EstimatedProbabilities[i] = malloc(lengthActionset * sizeof(double));

	for (int i = 0; i < maxStates; i++)
		for (int j = 0; j < lengthActionset; j++)
                EstimatedProbabilities[i][j] = -1;


QvaluesOne = malloc(maxStates * sizeof(double *));
	for(int i = 0; i < maxStates; i++)
		QvaluesOne[i] = malloc(lengthActionset * sizeof(double));

	for (int i = 0; i < maxStates; i++)
		for (int j = 0; j < lengthActionset; j++)
                QvaluesOne[i][j] = -1;

QvaluesTwo = malloc(maxStates * sizeof(double *));
	for(int i = 0; i < maxStates; i++)
		QvaluesTwo[i] = malloc(lengthActionset * sizeof(double));

	for (int i = 0; i < maxStates; i++)
		for (int j = 0; j < lengthActionset; j++)
                QvaluesTwo[i][j] = -1;

VisitsCount = malloc(maxStates * sizeof(int *));
	for(int i = 0; i < maxStates; i++)
		VisitsCount[i] = malloc(lengthActionset * sizeof(int));

	for (int i = 0; i < maxStates; i++)
		for (int j = 0; j < lengthActionset; j++)
                VisitsCount[i][j] = 0;

epsilonmapping = malloc(maxStates * sizeof(int));
    for(int i = 0; i < maxStates; i++){
        epsilonmapping[i] = malloc(maxAllowedActions * sizeof(int));
    }

	for (int i = 0; i < maxStates; i++)
		for (int j = 0; j < maxAllowedActions; j++)
            epsilonmapping[i][j] = 0;

previousmapping = malloc(maxStates * sizeof(int));
    for(int i = 0; i < maxStates; i++){
        previousmapping[i] = malloc(maxAllowedActions * sizeof(int));
    }

	for (int i = 0; i < maxStates; i++)
		for (int j = 0; j < maxAllowedActions; j++)
            previousmapping[i][j] = 0;

ActionStateProb = malloc(maxStates * sizeof(double *));
	for(int i = 0; i < maxStates; i++)
		ActionStateProb[i] = malloc(lengthActionset * sizeof(double));

	for (int i = 0; i < maxStates; i++)
		for (int j = 0; j < lengthActionset; j++)
                ActionStateProb[i][j] = -1;

batteryevo = malloc(runtimes * sizeof(double *));
	for(int i = 0; i < runtimes; i++)
		batteryevo[i] = malloc(epochs * sizeof(double));

	for (int i = 0; i < runtimes; i++)
		for (int j = 0; j < epochs; j++)
                batteryevo[i][j] = 0;

seclevelevo = malloc(runtimes * sizeof(int *));
	for(int i = 0; i < runtimes; i++)
		seclevelevo[i] = malloc(epochs * sizeof(int));

	for (int i = 0; i < runtimes; i++)
		for (int j = 0; j < epochs; j++)
                seclevelevo[i][j] = 0;

discardedpackets = malloc(runtimes * sizeof(double *));
	for(int i = 0; i < runtimes; i++)
		discardedpackets[i] = malloc(epochs * sizeof(double));

	for (int i = 0; i < runtimes; i++)
		for (int j = 0; j < epochs; j++)
                discardedpackets[i][j] = 0;

rewardsevo = malloc(runtimes * sizeof(double *));
	for(int i = 0; i < runtimes; i++)
		rewardsevo[i] = malloc(epochs * sizeof(double));

	for (int i = 0; i < runtimes; i++)
		for (int j = 0; j < epochs; j++)
                rewardsevo[i][j] = 0;

for (int idn = 0; idn <= packetsizemax; idn++)
    for (int icn = 0; icn < cnmax; icn++){
    /*   eprime(dn,1:cnmax)=[dn (dn+5) (dn+16) (dn+16+5)];*/
        if (idn==0)
            eprime[idn][icn]=0;
        else if( icn == 0 )
            overhead=0;
        else if( icn == 1 )
            overhead=5;
        else if( icn == 2 )
            overhead=16;
        else if( icn == 3 ){
            overhead=21;
        }

            if (idn!=0)
                eprime[idn][icn]=ceil((idn+overhead))/88*100;
            else
                eprime[idn][icn]=0;
    }

for (int idn = 0; idn <= packetsizemax; idn++)
    for (int icn = 0; icn < cnmax; icn++)
            printf("eprime[%d][%d]=%d",idn,icn,eprime[idn][icn]);


//readHarvesterAndPacketdata();

/*for (muhn=minenergy;muhn<=maxenergy;muhn++){
    for (int iepoch=0;iepoch<=epochs;iepoch++){
        printf("hnruntime[%d][%d]=%d",muhn,iepoch,hnruntime[muhn-minenergy][iepoch]);
    }
    for (int iepoch=0;iepoch<=epochs;iepoch++){
        printf("dnruntime[%d][%d]=%d",muhn,iepoch,dnruntime[muhn-minenergy][iepoch]);
    }
}*/


for (muhn=minenergy;muhn<=maxenergy;muhn++){   //minenergy-maxenergy

for(int alphaaux=8;alphaaux<=8;alphaaux=alphaaux+1){
    for(int betaaux=6;betaaux<=6;betaaux=betaaux+1){

/*--------------------------------------*/
/*-------Innitialisation starting-------*/
/*--------------------------------------*/
alpha=alphaaux/10;
beta=betaaux/10;
securityf[0]=0;
securityf[1]=1-alpha;
securityf[2]=alpha;
securityf[3]=1;


if (SARSA==1)
    InitialiseQValuesRandomly(maxStates,blockofstatesperlevel);

if (QLEARNING==1)
    InitialiseQValuesRandomly(maxStates,blockofstatesperlevel);

if (EXPECTEDSARSA==1)
    InitialiseQValuesRandomly(maxStates,blockofstatesperlevel);

if (DOUBLEQLEARNING==1)
    InitialiseQValuesRandomly(maxStates,blockofstatesperlevel);

if (NSTEPSARSA==1){
    InitialiseQValuesRandomly(maxStates,blockofstatesperlevel);
    deriveEpsilonGreedyPolicyfromQvalues(maxStates,blockofstatesperlevel);
}


/*------------------------------*/
/*-------episode starting-------*/
/*------------------------------*/
int integ;
double fract;

for(int episode=0;episode<episodes;episode++){

/*-------Episode preparation-------*/
iState=0;
while (iState==0)
    iState=gsl_rng_uniform_int(r,maxStates);    //Initialize S

cn=getcn(iState,blockofstatesperlevel);
bn=iState-(cn-1)*(bmax+1);

if (ENERGYVARIATION==1){
    if (episode==0)
        sigmahn=5;//muhn=65;
        sigmadn=3;
    if (episode==20)
        sigmahn=5;//muhn=65;
        sigmadn=1;
    if (episode==40)
        sigmahn=1;//muhn=65;
        sigmadn=1;
    if (episode==60)
        sigmahn=7;//muhn=65;
        sigmadn=7;
    if (episode==80)
        sigmahn=3;//muhn=65;
        sigmadn=3;
    minenergy=muhn;
    maxenergy=muhn;
}


for(int iepoch=0;iepoch<epochs;iepoch++){
    hnruntime[muhn-minenergy][iepoch]=gsl_ran_gaussian(r, sigmahn)+muhn;
    dnruntime[muhn-minenergy][iepoch]=gsl_ran_gaussian(r, sigmadn)+mudn;

// quantize hn
    integ=floor(hnruntime[muhn-minenergy][iepoch]);
    fract=hnruntime[muhn-minenergy][iepoch]-integ;
        if (hnruntime[muhn-minenergy][iepoch]<0.5)
            hnruntime[muhn-minenergy][iepoch]=0;
        else if (fract<0.5)
            hnruntime[muhn-minenergy][iepoch]=integ;
        else
            hnruntime[muhn-minenergy][iepoch]=ceil(hnruntime[muhn-minenergy][iepoch]);

// quantize dn
    integ=floor(dnruntime[muhn-minenergy][iepoch]);
    fract=dnruntime[muhn-minenergy][iepoch]-integ;
        if (dnruntime[muhn-minenergy][iepoch]<0.5)
            dnruntime[muhn-minenergy][iepoch]=0;
        else if (fract<0.5)
            dnruntime[muhn-minenergy][iepoch]=integ;
        else
            dnruntime[muhn-minenergy][iepoch]=ceil(dnruntime[muhn-minenergy][iepoch]);
}

/*----------Episode start----------*/

if (NAIVEPOLICY==0){

if (SARSA==1)
    deriveEpsilonGreedyPolicyfromQvalues(maxStates,blockofstatesperlevel);

//if (NSTEPSARSA==1)
  //  epochs=TnStepSARSA; //epoch = tau
//run internal simulation for average reward each episode after the first

/*alphaQlearning=alphaQlearning-(alphaQlearning/(episodes));
alphaSarsa=alphaSarsa-(alphaSarsa/(episodes));
alphaExpectedSARSA=alphaExpectedSARSA-(alphaExpectedSARSA/(episodes));
alphaDoubleQlearning=alphaDoubleQlearning-(alphaDoubleQlearning/(episodes));
alphanStepSARSA=alphanStepSARSA-(alphanStepSARSA/(episodes));*/


for(int epoch=0;epoch<epochs;epoch=epoch+1){


    //epsilon=epsilon-epsilon/(episodes*epochs);
    //printf("epsilon=%lf\n",epsilon);

    if (QLEARNING==1)
        deriveEpsilonGreedyPolicyfromQvalues(maxStates,blockofstatesperlevel);

    if (EXPECTEDSARSA==1)
        deriveEpsilonGreedyPolicyfromQvalues(maxStates,blockofstatesperlevel);

    if (DOUBLEQLEARNING==1)
        deriveEpsilonGreedyPolicyfromQOneTwovalues(maxStates,blockofstatesperlevel);

    currentState=iState;

    if ((NSTEPSARSA==1) && (epoch==0)){
            currentAction=selectRandomAction(currentState);
            stateVector[0]=currentState;
            actionVector[0]=currentAction;
    }
    else if (NSTEPSARSA==1)
        currentAction=actionVector[epoch];
    else
        currentAction=policy[currentState];

    lower=getlower(currentState,blockofstatesperlevel);
    upper=getupper(currentState,blockofstatesperlevel);

    VisitsCount[currentState][currentAction]=VisitsCount[currentState][currentAction]+1;

    nextcn=cn+currentAction-3;  //Take action A


    packetsize=dnruntime[muhn-minenergy][epoch];
    e=eprime[packetsize][cn-1];
    bn=bn+hnruntime[muhn-minenergy][epoch]-e;

    if (bn>bmax){
        bn=bmax;
        batteryratio=bn*1.0/bmax;
        Reward=(1-beta)*securityf[nextcn-1]+beta*(batteryratio); //observe R
    }
    else if (bn<=0){
        bn=iState-(cn-1)*(bmax+1);
        Reward=0; //observe R
    }
    else{
        batteryratio=bn*1.0/bmax;
        Reward=(1-beta)*securityf[nextcn-1]+beta*(batteryratio); //observe R
    }

    nextState=(bmax+1)*(nextcn-1)+bn; //observe S'

    if (SARSA==1)
        sampleAction(nextState);//deriveEpsilonGreedyPolicyfromQvalues(maxStates,blockofstatesperlevel);

    nextaction=policy[nextState]; // influences only SARSA Qvalues update

    if (NSTEPSARSA==1){
        stateVector[epoch+1]=nextState;
        rewardVector[epoch+1]=Reward;
        nextaction=selectRandomAction(nextState);
        actionVector[epoch+1]=nextaction;
        updateQValuesnStepSARSA(epoch,nstep,TnStepSARSA,gammanStepSARSA,alphanStepSARSA,maxStates,blockofstatesperlevel);
    }

    if (SARSA==1)
        updateQValuesSARSA(currentState,currentAction,nextState,nextaction,alphaSarsa,gammaSarsa,Reward,blockofstatesperlevel);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////UPDATE OK
    if (QLEARNING==1)
        updateQValuesQlearning(currentState,currentAction,nextState,alphaQlearning,gammaQlearning,Reward);

    if (EXPECTEDSARSA==1)
        updateQValuesExpectedSARSA(currentState,currentAction,nextState,alphaExpectedSARSA,gammaExpectedSARSA,Reward);

    if (DOUBLEQLEARNING==1)
        updateQValuesDoubleQlearning(currentState,currentAction,nextState,alphaDoubleQlearning,gammaDoubleQlearning,Reward);

    //Next State
    cn=nextcn;
    iState=nextState;

    if (ACTORCRITIC==1)
        I=gammaACTORCRITIC*I;

        //printf("I am at epoch %d\n",epoch);
}//epochs ending

printf("Episode %d ending for muhn=%d \n",episode,muhn);
//if (episode>1400)
  //  printf("rewards are too low");
    //run internal simulation after Q values update
    simulatePolicy(bmax,alpha,beta);

}else{
    printf("Episode %d ending for muhn=%d \n",episode,muhn);
    simulatePolicy(bmax,alphaaux/10.0,betaaux/10.0);
}

}// episode ending

// Extracts deterministic greedy policy from Qvalues (works for SARSA, EcpectedSARSA and Qlearning)
if ((SARSA==1) || (QLEARNING==1) || (EXPECTEDSARSA==1) || (NSTEPSARSA==1))
    ExtractDeterministicPolicyfromQvalues(maxStates);

if (DOUBLEQLEARNING==1)
    ExtractDeterministicPolicyfromQOneTwovalues(maxStates);

//if (ACTORCRITIC==1)
  //  ExtractStochasticPolicyfromPreferences(maxStates,blockofstatesperlevel,bmax,alpha,beta);

if (SARSA==1)
    PrintSarsaPolicyFile(maxStates,blockofstatesperlevel);

if (QLEARNING==1)
    PrintQlearningPolicyFile(maxStates,blockofstatesperlevel);

if (EXPECTEDSARSA==1)
    PrintExpectedSarsaPolicyFile(maxStates,blockofstatesperlevel);

if (DOUBLEQLEARNING==1)
    PrintDoubleQlearningPolicyFile(maxStates,blockofstatesperlevel);

if (NSTEPSARSA==1)
    PrintnStepSARSAPolicyFile(maxStates,blockofstatesperlevel);

if (ACTORCRITIC==1)
    PrintACTORCRITICPolicyFile(maxStates,blockofstatesperlevel,bmax,alpha,beta);

if (SIMULATION==1){
    simulatePolicy(bmax,alpha,beta);

}

} //end alpha
} //end beta

} //end muhn

return (EXIT_SUCCESS);

}//main ending


int getlower(int i, int statesdivision){
        if (i<statesdivision)
            lower=3;
        else if ((i>=statesdivision) && (i<2*statesdivision))
            lower=2;
        else if ((i>=2*statesdivision) && (i<3*statesdivision))
            lower=1;
        else if (i>=3*statesdivision)
            lower=0;
    return lower;
}

int getupper(int i, int statesdivision){
        if (i<statesdivision)
            upper=lengthActionset-1;
        else if ((i>=statesdivision) && (i<2*statesdivision))
            upper=lengthActionset-2;
        else if ((i>=2*statesdivision) && (i<3*statesdivision))
            upper=lengthActionset-3;
        else if (i>=3*statesdivision)
            upper=lengthActionset-4;
    return upper;
}

int getcn(int i, int statesdivision){
int cn;
        if (i<statesdivision)
            cn=1;
        else if ((i>=statesdivision) && (i<2*statesdivision))
            cn=2;
        else if ((i>=2*statesdivision) && (i<3*statesdivision))
            cn=3;
        else if (i>=3*statesdivision)
            cn=4;
    return cn;
}


void PrintSarsaPolicyFile (int nStates, int statesdivision){
FILE *fileSarsa;
fileSarsa=fopen("SARSA policy.txt","a");

    for (int i=0;i<nStates;i=i+1){
        int rem=i%10;
        if ((rem==0) && (i!=1)){
            fprintf(fileSarsa,"\n");
        }
        fprintf(fileSarsa,"policy[%d]=%d ",i,policy[i]);
    }

fprintf(fileSarsa,"\n\n\n");

    for (int i = 0; i < nStates; i++)
		for (int j = 0; j < lengthActionset; j++){
            if (Qvalues[i][j]!=-1)
                fprintf(fileSarsa,"Qvalues[%d][%d]=%lf ",i,j,Qvalues[i][j]);
		}

fprintf(fileSarsa,"\n\n\n");

    for (int i = 0; i < nStates; i++){
        if (i<statesdivision){
            upper=lengthActionset-1;
            lower=3;
        }
        else if ((i>=statesdivision) && (i<2*statesdivision)){
            upper=lengthActionset-2;
            lower=2;
        }
        else if ((i>=2*statesdivision) && (i<3*statesdivision)){
            upper=lengthActionset-3;
            lower=1;
        }
        else if (i>=3*statesdivision){
            upper=lengthActionset-4;
            lower=0;
        }
 		for (int j = lower; j <= upper; j++){
            if (VisitsCount[i][j]!=-1)
                fprintf(fileSarsa,"VisitsCount[%d][%d]=%d ",i,j,VisitsCount[i][j]);
		}
    }

fprintf(fileSarsa,"\n");

fclose(fileSarsa);
}

void PrintQlearningPolicyFile (int nStates, int statesdivision){
FILE *fileQlearning;
fileQlearning=fopen("Qlearning policy.txt","a");

    for (int i=0;i<nStates;i=i+1){
        int rem=i%10;
        if ((rem==0) && (i!=1)){
            fprintf(fileQlearning,"\n");
        }
        fprintf(fileQlearning,"policy[%d]=%d ",i,policy[i]);
    }

fprintf(fileQlearning,"\n\n\n");

    for (int i = 0; i < nStates; i++)
		for (int j = 0; j < lengthActionset; j++){
            if (Qvalues[i][j]!=-1)
                fprintf(fileQlearning,"Qvalues[%d][%d]=%lf ",i,j,Qvalues[i][j]);
		}

fprintf(fileQlearning,"\n\n\n");

    for (int i = 0; i < nStates; i++){
        if (i<statesdivision){
            upper=lengthActionset-1;
            lower=3;
        }
        else if ((i>=statesdivision) && (i<2*statesdivision)){
            upper=lengthActionset-2;
            lower=2;
        }
        else if ((i>=2*statesdivision) && (i<3*statesdivision)){
            upper=lengthActionset-3;
            lower=1;
        }
        else if (i>=3*statesdivision){
            upper=lengthActionset-4;
            lower=0;
        }
 		for (int j = lower; j <= upper; j++){
            if (VisitsCount[i][j]!=-1)
                fprintf(fileQlearning,"VisitsCount[%d][%d]=%d ",i,j,VisitsCount[i][j]);
		}
    }

fprintf(fileQlearning,"\n");


fclose(fileQlearning);
}

void PrintExpectedSarsaPolicyFile (int nStates, int statesdivision){
FILE *fileExpectedSarsa;
fileExpectedSarsa=fopen("ExpectedSarsa policy.txt","a");

    for (int i=0;i<nStates;i=i+1){
        int rem=i%10;
        if ((rem==0) && (i!=1)){
            fprintf(fileExpectedSarsa,"\n");
        }
        fprintf(fileExpectedSarsa,"policy[%d]=%d ",i,policy[i]);
    }

fprintf(fileExpectedSarsa,"\n\n\n");

    for (int i = 0; i < nStates; i++)
		for (int j = 0; j < lengthActionset; j++){
            if (Qvalues[i][j]!=-1)
                fprintf(fileExpectedSarsa,"Qvalues[%d][%d]=%lf ",i,j,Qvalues[i][j]);
		}

fprintf(fileExpectedSarsa,"\n\n\n");

    for (int i = 0; i < nStates; i++){
        if (i<statesdivision){
            upper=lengthActionset-1;
            lower=3;
        }
        else if ((i>=statesdivision) && (i<2*statesdivision)){
            upper=lengthActionset-2;
            lower=2;
        }
        else if ((i>=2*statesdivision) && (i<3*statesdivision)){
            upper=lengthActionset-3;
            lower=1;
        }
        else if (i>=3*statesdivision){
            upper=lengthActionset-4;
            lower=0;
        }
 		for (int j = lower; j <= upper; j++){
            if (VisitsCount[i][j]!=-1)
                fprintf(fileExpectedSarsa,"VisitsCount[%d][%d]=%d ",i,j,VisitsCount[i][j]);
		}
    }

fprintf(fileExpectedSarsa,"\n");


fclose(fileExpectedSarsa);

}

void PrintDoubleQlearningPolicyFile (int nStates, int statesdivision){
FILE *fileDoubleQlearning;
fileDoubleQlearning=fopen("DoubleQlearning policy.txt","a");

    for (int i=0;i<nStates;i=i+1){
        int rem=i%10;
        if ((rem==0) && (i!=1)){
            fprintf(fileDoubleQlearning,"\n");
        }
        fprintf(fileDoubleQlearning,"policy[%d]=%d ",i,policy[i]);
    }

fprintf(fileDoubleQlearning,"\n\n\n");

    for (int i = 0; i < nStates; i++)
		for (int j = 0; j < lengthActionset; j++){
            if (Qvalues[i][j]!=-1)
                fprintf(fileDoubleQlearning,"Qvalues[%d][%d]=%lf ",i,j,Qvalues[i][j]);
		}

fprintf(fileDoubleQlearning,"\n\n\n");

    for (int i = 0; i < nStates; i++){
        if (i<statesdivision){
            upper=lengthActionset-1;
            lower=3;
        }
        else if ((i>=statesdivision) && (i<2*statesdivision)){
            upper=lengthActionset-2;
            lower=2;
        }
        else if ((i>=2*statesdivision) && (i<3*statesdivision)){
            upper=lengthActionset-3;
            lower=1;
        }
        else if (i>=3*statesdivision){
            upper=lengthActionset-4;
            lower=0;
        }
 		for (int j = lower; j <= upper; j++){
            if (VisitsCount[i][j]!=-1)
                fprintf(fileDoubleQlearning,"VisitsCount[%d][%d]=%d ",i,j,VisitsCount[i][j]);
		}
    }

fprintf(fileDoubleQlearning,"\n");


fclose(fileDoubleQlearning);

}

void PrintnStepSARSAPolicyFile (int nStates, int statesdivision){
FILE *filenStepSARSA;
filenStepSARSA=fopen("nStepSARSA policy.txt","a");

    for (int i=0;i<nStates;i=i+1){
        int rem=i%10;
        if ((rem==0) && (i!=1)){
            fprintf(filenStepSARSA,"\n");
        }
        fprintf(filenStepSARSA,"policy[%d]=%d ",i,policy[i]);
    }

fprintf(filenStepSARSA,"\n\n\n");

    for (int i = 0; i < nStates; i++)
		for (int j = 0; j < lengthActionset; j++){
            if (Qvalues[i][j]!=-1)
                fprintf(filenStepSARSA,"Qvalues[%d][%d]=%lf ",i,j,Qvalues[i][j]);
		}

fprintf(filenStepSARSA,"\n\n\n");

    for (int i = 0; i < nStates; i++){
        if (i<statesdivision){
            upper=lengthActionset-1;
            lower=3;
        }
        else if ((i>=statesdivision) && (i<2*statesdivision)){
            upper=lengthActionset-2;
            lower=2;
        }
        else if ((i>=2*statesdivision) && (i<3*statesdivision)){
            upper=lengthActionset-3;
            lower=1;
        }
        else if (i>=3*statesdivision){
            upper=lengthActionset-4;
            lower=0;
        }
 		for (int j = lower; j <= upper; j++){
            if (VisitsCount[i][j]!=-1)
                fprintf(filenStepSARSA,"VisitsCount[%d][%d]=%d ",i,j,VisitsCount[i][j]);
		}
    }

fprintf(filenStepSARSA,"\n");


fclose(filenStepSARSA);

}

int PrintACTORCRITICPolicyFile(int maxStates,int statesdivision, int bmax, double alpha, double beta){
FILE *fileACTORCRITIC;
fileACTORCRITIC=fopen("ACTOR-CRITIC policy.txt","a");

    for (int istate=0;istate<maxStates;istate=istate+1){
        int rem=istate%10;
        if ((rem==0) && (istate!=1)){
            fprintf(fileACTORCRITIC,"\n");
        }

        ExtractStochasticPolicyfromPreferences(istate,statesdivision,bmax,alpha,beta);

        if (istate<statesdivision){
            upper=lengthActionset-1;
            lower=3;
        }
        else if ((istate>=statesdivision) && (istate<2*statesdivision)){
            upper=lengthActionset-2;
            lower=2;
        }
        else if ((istate>=2*statesdivision) && (istate<3*statesdivision)){
            upper=lengthActionset-3;
            lower=1;
        }
        else if (istate>=3*statesdivision){
            upper=lengthActionset-4;
            lower=0;
        }

        for (int iaction=0;iaction<maxAllowedActions;iaction++){
            fprintf(fileACTORCRITIC,"policy[%d]=%lf ",iaction+lower,preferencesmapping[iaction]);
        }
    }
        return 1;
}

int InitialiseQValuesRandomly(int nStates, int statesdivision){

/*----------------------Arbitrary Q values innit---------------------*/

for (int state=0;state<nStates;state++){
    if (state<statesdivision){
        upper=lengthActionset-1;
        lower=3;
    }
    else if ((state>=statesdivision) && (state<2*statesdivision)){
        upper=lengthActionset-2;
        lower=2;
    }
    else if ((state>=2*statesdivision) && (state<3*statesdivision)){
        upper=lengthActionset-3;
        lower=1;
    }
    else if (state>=3*statesdivision){
        upper=lengthActionset-4;
        lower=0;
    }
    for (int iaction=lower;iaction<=upper;iaction++){
        Qvalues[state][iaction]=gsl_rng_uniform(r);
        QvaluesOne[state][iaction]=0;
        QvaluesTwo[state][iaction]=0;
    }
}

return 1;
}

int InitialiseEstimatedQValuesfromRandomWeights(int nStates, int statesdivision, int bmax, double alpha, double beta){
int bn,cn;

    mapWeights(ANNWeightsold);

    for(int istate=0;istate<nStates;istate++){
        if (istate<statesdivision)
            cn=1;
        else if ((istate>=statesdivision) && (istate<2*statesdivision))
            cn=2;
        else if ((istate>=2*statesdivision) && (istate<3*statesdivision))
            cn=3;
        else if (istate>=3*statesdivision)
            cn=4;

        bn=istate-(bmax+1)*(cn-1);

        inputLayerNeurons[0]=bn;
        inputLayerNeurons[1]=cn;
        //inputLayerNeurons[2]=alpha;
        //inputLayerNeurons[3]=beta;

        ForwardPropagate();

        for (int iaction=0;iaction<lengthActionset;iaction++){
            EstimatedQvalues[istate][iaction]=outputLayerNeurons[iaction];
            printf("EstimatedQvalues[%d][%d]=%lf",istate,iaction,EstimatedQvalues[istate][iaction]);
        }
    }

return 1;
}

int InitialiseEstimatedProbabilitiesfromRandomWeights(int nStates, int statesdivision, int bmax, double alpha, double beta){
int bn,cn;

    mapWeights(probWeightsold);

    for(int istate=0;istate<nStates;istate++){
        bn=istate-(bmax+1)*(cn-1);
        cn=(istate-bn)/(bmax+1)-1;

        inputLayerNeurons[0]=bn;
        inputLayerNeurons[1]=cn;
        //inputLayerNeurons[2]=alpha;
        //inputLayerNeurons[3]=beta;

        ForwardPropagate();

    for (int iaction=0;iaction<=lengthActionset;iaction++)
        preferences[iaction]=outputLayerNeurons[iaction];
    }

return 1;
}

int deriveEpsilonGreedyPolicyfromQvalues(int nStates, int statesdivision){
    double Qstar=-2;
    int counter, Astar,epsilonmappingoffset;
    double rndnumber;
    double epsilonprime=epsilon/4.0;

/*---------Generate policy from Q values in an epsilon-greedy fashion-----------*/

for (int state=0;state<nStates;state++){
    Qstar=-2;
    Astar=-1;
    counter=0;

    // Define possible actions
        epsilonmappingoffset=0;
        if (state<statesdivision){
            upper=lengthActionset-1;
            lower=3;
        }
        else if ((state>=statesdivision) && (state<2*statesdivision)){
            upper=lengthActionset-2;
            lower=2;
        }
        else if ((state>=2*statesdivision) && (state<3*statesdivision)){
            upper=lengthActionset-3;
            lower=1;
        }
        else if (state>=3*statesdivision){
            upper=lengthActionset-4;
            lower=0;
        }
        maxAllowedActions=upper-lower+1;

    // go to all possible actions and check highest Qvalue
    for (int actionaux=lower;actionaux<=upper;actionaux++){
        if (Qvalues[state][actionaux]>Qstar){
            Astar=actionaux;
            Qstar=Qvalues[state][actionaux];
            counter=1;
        } else if (Qvalues[state][actionaux]==Qstar)
            counter++;
    }

    // break ties at random among optimal actions
        if (counter>1) {
            rndnumber=rand()%(counter-1) + 1;
                for (int actionaux=lower;actionaux<=upper;actionaux++){
                    if (Qvalues[state][actionaux]==Qstar) {
                        rndnumber--;
                            if (rndnumber==0)
                                Astar=actionaux;
                    }
                }
        }

        for (int iactionprob=lower;iactionprob<=upper;iactionprob++){
            if (iactionprob==Astar){
                epsilonmapping[state][0]=iactionprob;
                ActionStateProb[state][iactionprob]=1-epsilon+(epsilon/maxAllowedActions);
                //printf("ActionStateProb[%d][%d]=%lf ",state,iactionprob,ActionStateProb[state][iactionprob]);
            }
            else{
                epsilonmappingoffset++;
                epsilonmapping[state][epsilonmappingoffset]=iactionprob;
                ActionStateProb[state][iactionprob]=epsilon/maxAllowedActions;
                //printf("ActionStateProb[%d][%d]=%lf ",state,iactionprob,ActionStateProb[state][iactionprob]);
            }
        }

        rndnumber=gsl_rng_uniform(r);

        if (rndnumber<epsilonprime)
            policy[state]=epsilonmapping[state][1];
        else if ((rndnumber>=epsilonprime) && (rndnumber<2.0*epsilonprime))
            policy[state]=epsilonmapping[state][2];
        else if ((rndnumber>=2.0*epsilonprime) && (rndnumber<3.0*epsilonprime))
            policy[state]=epsilonmapping[state][3];
        else
            policy[state]=epsilonmapping[state][0];   // optimal action Astar

}

return 1;

}

int deriveEpsilonGreedyPolicyfromQOneTwovalues(int nStates, int statesdivision){
    double Qstar=-2;
    int counter, Astar,epsilonmappingoffset;
    double rndnumber;
    double epsilonprime=epsilon/4.0;

/*---------Generate policy from Q values in an epsilon-greedy fashion-----------*/

for (int state=0;state<nStates;state++){
    Qstar=-2;
    Astar=-1;
    counter=0;

    // Define possible actions
        epsilonmappingoffset=0;
        if (state<statesdivision){
            upper=lengthActionset-1;
            lower=3;
        }
        else if ((state>=statesdivision) && (state<2*statesdivision)){
            upper=lengthActionset-2;
            lower=2;
        }
        else if ((state>=2*statesdivision) && (state<3*statesdivision)){
            upper=lengthActionset-3;
            lower=1;
        }
        else if (state>=3*statesdivision){
            upper=lengthActionset-4;
            lower=0;
        }
        maxAllowedActions=upper-lower+1;

    // go to all possible actions and check highest Qvalue
    for (int actionaux=lower;actionaux<=upper;actionaux++){
        if (QvaluesOne[state][actionaux]+QvaluesTwo[state][actionaux]>Qstar){
            Astar=actionaux;
            Qstar=QvaluesOne[state][actionaux]+QvaluesTwo[state][actionaux];
            counter=1;
        } else if (QvaluesOne[state][actionaux]+QvaluesTwo[state][actionaux]==Qstar)
            counter++;
    }

    // break ties at random among optimal actions
        if (counter>1) {
            rndnumber=rand()%(counter-1) + 1;
                for (int actionaux=lower;actionaux<=upper;actionaux++){
                    if (QvaluesOne[state][actionaux]+QvaluesTwo[state][actionaux]==Qstar) {
                        rndnumber--;
                            if (rndnumber==0)
                                Astar=actionaux;
                    }
                }
        }

        for (int iactionprob=lower;iactionprob<=upper;iactionprob++){
            if (iactionprob==Astar){
                epsilonmapping[state][0]=iactionprob;
                ActionStateProb[state][iactionprob]=1-epsilon+(epsilon/maxAllowedActions);
                //printf("ActionStateProb[%d][%d]=%lf ",state,iactionprob,ActionStateProb[state][iactionprob]);
            }
            else{
                epsilonmappingoffset++;
                epsilonmapping[state][epsilonmappingoffset]=iactionprob;
                ActionStateProb[state][iactionprob]=epsilon/maxAllowedActions;
                //printf("ActionStateProb[%d][%d]=%lf ",state,iactionprob,ActionStateProb[state][iactionprob]);
            }
        }

        rndnumber=gsl_rng_uniform(r);

        if (rndnumber<epsilonprime)
            policy[state]=epsilonmapping[state][1];
        else if ((rndnumber>=epsilonprime) && (rndnumber<2.0*epsilonprime))
            policy[state]=epsilonmapping[state][2];
        else if ((rndnumber>=2.0*epsilonprime) && (rndnumber<3.0*epsilonprime))
            policy[state]=epsilonmapping[state][3];
        else
            policy[state]=epsilonmapping[state][0];   // optimal action Astar

}

return 1;

}

int sampleAction(int state){
double epsilonprime=epsilon/4.0;
double rndnumber;

    rndnumber=gsl_rng_uniform(r);

    if (rndnumber<epsilonprime)
        policy[state]=epsilonmapping[state][1];
    else if ((rndnumber>=epsilonprime) && (rndnumber<2.0*epsilonprime))
        policy[state]=epsilonmapping[state][2];
    else if ((rndnumber>=2.0*epsilonprime) && (rndnumber<3.0*epsilonprime))
        policy[state]=epsilonmapping[state][3];
    else
        policy[state]=epsilonmapping[state][0];   // optimal action Astar

    return 1;
}


int ExtractDeterministicPolicyfromQvalues(int nStates){
  //  double Qvalueaux=-2;

/*-----------Extract policy from Q values-----------*/

for (int i=0;i<nStates;i++){
        policy[i]=epsilonmapping[i][0];
/*
Qvalueaux=-2;
    for (int iactionQ=0;iactionQ<lengthActionset;iactionQ++)
        if (Qvalues[i][iactionQ]>Qvalueaux){
            Qvalueaux=Qvalues[i][iactionQ];
            policy[i]=iactionQ;
        }
*/
}

return 1;
}

int ExtractStochasticPolicyfromPreferences(int istate,int statesdivision, int bmax, double alpha, double beta){
int lower,upper,cn,bn;

    if (istate<statesdivision){
        upper=lengthActionset-1;
        lower=3;
        cn=1;
    }
    else if ((istate>=statesdivision) && (istate<2*statesdivision)){
        upper=lengthActionset-2;
        lower=2;
        cn=2;
    }
    else if ((istate>=2*statesdivision) && (istate<3*statesdivision)){
        upper=lengthActionset-3;
        lower=1;
        cn=3;
    }
    else if (istate>=3*statesdivision){
        upper=lengthActionset-4;
        lower=0;
        cn=4;
    }
        bn=istate-(bmax+1)*(cn-1);

        mapWeights(probWeightsnew);
        inputLayerNeurons[0]=bn;
        inputLayerNeurons[1]=cn;
        //inputLayerNeurons[2]=alpha;
        //inputLayerNeurons[3]=beta;
        ForwardPropagate();

        for (int i=0;i<numOutputNeurons;i++)
            preferences[i]=outputLayerNeurons[i];

        softMaxProbabilities(preferences,lengthActionset,istate,statesdivision,lower,upper);   // preferences are now probability values
                                                                                              // preferencesmapping are now allowed actions' probability values
        //probabilities for each allowed state are now stored in preferencesmapping

    return 1;
}

int ExtractDeterministicPolicyfromQOneTwovalues(int nStates){
    double Qstar=-2;

/*-----------Extract policy from Q values-----------*/

for (int state=0;state<nStates;state++){
      //  policy[i]=epsilonmapping[i][0];
Qstar=-2;
    for (int iactionQ=0;iactionQ<lengthActionset;iactionQ++)
        if (QvaluesOne[state][iactionQ]+QvaluesTwo[state][iactionQ]>Qstar){
            Qstar=QvaluesOne[state][iactionQ]+QvaluesTwo[state][iactionQ];
            policy[state]=iactionQ;
        }
}

return 1;
}

int updateQValuesSARSA(int curState, int curAction, int nState, int nAction, double aSarsa, double gSarsa, double R, int statesdivision){
    Qvalues[curState][curAction]=Qvalues[curState][curAction]+aSarsa*(R+gSarsa*Qvalues[nState][nAction]-Qvalues[curState][curAction]);
    //Qvalues[curState][curAction]=Qvalues[curState][curAction]+aSarsa*(R+gSarsa*Qvalues[nState][nAction]-Qvalues[curState][curAction])*serror;
    //printf("Qvalues[%d][%d]=%lf",curState,curAction,Qvalues[curState][curAction]);

return 1;
}

int updateQValuesQlearning(int curState, int curAction, int nState, double aQlearning, double gQlearning, double R){
int Astar=-1;
double Qstar=-1;

    for (int actionaux=0;actionaux<lengthActionset;actionaux++){
        if (Qvalues[nState][actionaux]>Qstar)
            Astar=actionaux;
    }
    //printf("Qvalues[%d][%d]=%lf",curState,curAction,Qvalues[curState][curAction]);
    Qvalues[curState][curAction]=Qvalues[curState][curAction]+aQlearning*(R+gQlearning*Qvalues[nState][Astar]-Qvalues[curState][curAction]);
    if (Qvalues[curState][curAction]<0)
        printf("fodeu");
    //Qvalues[curState][curAction]=Qvalues[curState][curAction]+aQlearning*(R+gQlearning*Qvalues[nState][Astar]-Qvalues[curState][curAction])*serror;
    //printf("Qvalues[%d][%d]=%lf",curState,curAction,Qvalues[curState][curAction]);

return 1;
}

int updateQValuesExpectedSARSA(int curState, int curAction, int nState, double aExpectedSARSA, double gExpectedSARSA, double R){
double ExpectedValue;

ExpectedValue=0;

for (int iaction=0;iaction<lengthActionset;iaction++){
    if (ActionStateProb[nState][iaction]>0)
        ExpectedValue=ExpectedValue+ActionStateProb[nState][iaction]*Qvalues[nState][iaction];
}

    Qvalues[curState][curAction]=Qvalues[curState][curAction]+aExpectedSARSA*(R+gExpectedSARSA*ExpectedValue-Qvalues[curState][curAction]);

return 1;
}

int updateQValuesDoubleQlearning(int curState, int curAction, int nState, double aDoubleQlearning, double gDoubleQlearning, double R){
double argmaxA;
int Astar;
double rndnumber;

argmaxA=-2;
Astar=-1;

rndnumber=gsl_rng_uniform(r);

    if (rndnumber<0.5){    //UPDATE Q1
        for (int iaction=0;iaction<lengthActionset;iaction++){
            if (QvaluesOne[nState][iaction]>argmaxA){
                argmaxA=QvaluesOne[nState][iaction];
                Astar=iaction;
            }
        }
    QvaluesOne[curState][curAction]=QvaluesOne[curState][curAction]+aDoubleQlearning*(R+gDoubleQlearning*QvaluesTwo[nState][Astar]-QvaluesOne[curState][curAction]);
    }
    else if (rndnumber>=0.5){                   //UPDATE Q2
        for (int iaction=0;iaction<lengthActionset;iaction++){
            if (QvaluesTwo[nState][iaction]>argmaxA){
                argmaxA=QvaluesTwo[nState][iaction];
                Astar=iaction;
            }
        }
    QvaluesTwo[curState][curAction]=QvaluesTwo[curState][curAction]+aDoubleQlearning*(R+gDoubleQlearning*QvaluesOne[nState][Astar]-QvaluesTwo[curState][curAction]);
    }

return 1;
}

int updateQValuesnStepSARSA(int epoch, int nstep, int TnStepSARSA, double gammanStepSARSA, double alphanStepSARSA, int maxStates, int blockofstatesperlevel){
int tau;
int iauxmax;
double G;

        tau=epoch-nstep+1;
        if (tau+nstep<=TnStepSARSA)
            iauxmax=tau+nstep;
        else
            iauxmax=TnStepSARSA;
        G=0;
        if (tau>=0){
            for(int i=tau+1;i<=iauxmax;i++){
                G=G+pow(gammanStepSARSA,i-tau-1)*rewardVector[i];
            }
            if(tau+nstep<TnStepSARSA){
                int stateaux=stateVector[tau+nstep];
                int actionaux=actionVector[tau+nstep];
                double power=pow(gammanStepSARSA,nstep);
                G=G+power*Qvalues[stateaux][actionaux];
            }
            Qvalues[stateVector[tau]][actionVector[tau]]=Qvalues[stateVector[tau]][actionVector[tau]]+alphanStepSARSA*(G-Qvalues[stateVector[tau]][actionVector[tau]]);
            deriveEpsilonGreedyPolicyfromQvalues(maxStates,blockofstatesperlevel);
        }
return 1;
}

int selectRandomAction(int state){
double rndnumber;
double epsilonprime=epsilon/4.0;
int currentAction;

    rndnumber=gsl_rng_uniform(r);

    if (rndnumber<epsilonprime)
        currentAction=epsilonmapping[state][1];
    else if ((rndnumber>=epsilonprime) && (rndnumber<2.0*epsilonprime))
        currentAction=epsilonmapping[state][2];
    else if ((rndnumber>=2.0*epsilonprime) && (rndnumber<3.0*epsilonprime))
        currentAction=epsilonmapping[state][3];
    else
        currentAction=epsilonmapping[state][0];

    return currentAction;
}

int selectRandomActionACTORCRITIC(int state, int statesdivision){
double rndnumber;
int currentAction;
int lower,upper;

        if (state<statesdivision){
            upper=lengthActionset-1;
            lower=3;
        }
        else if ((state>=statesdivision) && (state<2*statesdivision)){
            upper=lengthActionset-2;
            lower=2;
        }
        else if ((state>=2*statesdivision) && (state<3*statesdivision)){
            upper=lengthActionset-3;
            lower=1;
        }
        else if (state>=3*statesdivision){
            upper=lengthActionset-4;
            lower=0;
        }

        mapWeights(probWeightsnew);
        ForwardPropagate();

        for (int i=0;i<numOutputNeurons;i++)
            preferences[i]=outputLayerNeurons[i];

        softMaxProbabilities(preferences,lengthActionset,state,statesdivision,lower,upper);   // preferences are now probability values
                                                                                              // preferencesmapping are now allowed actions' probability values
        rndnumber=gsl_rng_uniform(r);

        if (rndnumber<=preferencesmapping[0])
            currentAction=lower;
        else if ((rndnumber>preferencesmapping[0]) && (rndnumber<=preferencesmapping[0]+preferencesmapping[1]))
            currentAction=lower+1;
        else if ((rndnumber>preferencesmapping[0]+preferencesmapping[1]) && (rndnumber<preferencesmapping[0]+preferencesmapping[1]+preferencesmapping[2]))
            currentAction=lower+2;
        else
            currentAction=upper;

        return currentAction;
}

double* softMaxProbabilities(double *logits, int inputsize, int curstate, int statesdivision, int lower, int upper){
double sumProbs = 0.0, probLogit;
int counter=-1;

    for (int iLogit = 0; iLogit < inputsize; iLogit++) {
        if ((iLogit>=lower) && (iLogit<=upper))
            sumProbs = sumProbs + exp(logits[iLogit]);
    }

    for (int iLogit = 0; iLogit < inputsize; iLogit++) {
        if ((iLogit>=lower) && (iLogit<=upper)){
            counter++;
            probLogit = exp(logits[iLogit]);
            logits[iLogit]=probLogit/sumProbs;
            preferencesmapping[counter]=logits[iLogit];
        }
        else
            logits[iLogit]=-1;
    }

        for (int iLogit = 0; iLogit < inputsize; iLogit++)
            printf("logits[%d]=%lf",iLogit,logits[iLogit]);
        for (int iLogit = 0; iLogit < maxAllowedActions; iLogit++)
            printf("preferencesmapping[%d]=%lf",iLogit,preferencesmapping[iLogit]);

return logits;

}

int updateParametersACTORCRITIC(int curState,int nState,int curaction, int naction, int cn, double R, int bmax, double alpha, double beta, int statesdivision,double gACTORCRITIC,int numWeights,double alphaW, double alphaT){
double delta=0;
double actionvalueNextState=0;
double actionvalueCurrentState=0;
double probActionCurrentState=0;
int nextcn, upper, lower;

    if (curState<statesdivision){
        upper=lengthActionset-1;
        lower=3;
    }
    else if ((curState>=statesdivision) && (curState<2*statesdivision)){
        upper=lengthActionset-2;
        lower=2;
    }
    else if ((curState>=2*statesdivision) && (curState<3*statesdivision)){
        upper=lengthActionset-3;
        lower=1;
    }
    else if (curState>=3*statesdivision){
        upper=lengthActionset-4;
        lower=0;
    }

    mapWeights(ANNWeightsnew);
    actionvalueCurrentState=estimateActionValue(curState,curaction,cn,alpha,beta,statesdivision,bmax);
    nextcn=cn+curaction-3;
    actionvalueNextState=estimateActionValue(nState,naction,nextcn,alpha,beta,statesdivision,bmax);

    updateDerivativeQ(curState,curaction,numWeights,actionvalueCurrentState);
    //derivativeQ
    delta=R+gACTORCRITIC*actionvalueNextState-actionvalueCurrentState;

    //updateDerivativeQ(curState,curaction,numWeights,actionvalueCurrentState);
    updateWeightsActionValueFunction(curState,delta,alphaW);

    mapWeights(probWeightsnew);

    probActionCurrentState=estimateActionProbability(curState,curaction,cn,alpha,beta,statesdivision,bmax,lower,upper);
    updateDerivativeProb(curState,curaction,numWeights,probActionCurrentState);
    updateWeightsActionProbabilityFunction(numWeights,delta,alphaT);

    return 1;
}

int updateDerivativeProb(int curState,int curaction, int numWeights, double probActionCurrentState){
    for (int iWeight=0;iWeight<numWeights;iWeight++){
        printf("derivativeProb[%d]=%lf",iWeight,derivativeProb[iWeight]);
        derivativeProb[iWeight]=(probActionCurrentState-EstimatedProbabilities[curState][curaction])/(probWeightsnew[iWeight]-probWeightsold[iWeight]);
        printf("derivativeProb[%d]=%lf",iWeight,derivativeProb[iWeight]);
    }
        return 1;
}

int updateDerivativeQ(int curState,int curaction, int numWeights,double actionvalueCurrentState){
    for (int iWeight=0;iWeight<numWeights;iWeight++){
        printf("derivativeQ[%d]=%lf",iWeight,derivativeQ[iWeight]);
        derivativeQ[iWeight]=(actionvalueCurrentState-EstimatedQvalues[curState][curaction])/(ANNWeightsnew[iWeight]-ANNWeightsold[iWeight]);
        printf("derivativeQ[%d]=%lf",iWeight,derivativeQ[iWeight]);
    }
        return 1;
}

int updateWeightsActionValueFunction(int numWeights, double delta, double alphaW){
    for (int iWeight=0;iWeight<numWeights;iWeight++){
        printf("ANNWeightsnew[%d]=%lf",iWeight,ANNWeightsnew[iWeight]);
        ANNWeightsold[iWeight]=ANNWeightsnew[iWeight];
        printf("alphaWeights*delta*derivativeQ[%d]=%lf",iWeight,alphaW*delta*derivativeQ[iWeight]);
        ANNWeightsnew[iWeight]=ANNWeightsnew[iWeight]+alphaW*delta*derivativeQ[iWeight];
        printf("ANNWeightsnew[%d]=%lf",iWeight,ANNWeightsnew[iWeight]);
    }
    return 1;
}

int updateWeightsActionProbabilityFunction(int numWeights, double delta, double alphaTheta){
    for (int iWeight=0;iWeight<numWeights;iWeight++){
        probWeightsold[iWeight]=probWeightsnew[iWeight];
        printf("probWeightsnew[%d]=%lf",iWeight,probWeightsnew[iWeight]);
        probWeightsnew[iWeight]=probWeightsnew[iWeight]+alphaTheta*delta*derivativeProb[iWeight];
        printf("probWeightsnew[%d]=%lf",iWeight,probWeightsnew[iWeight]);
    }
    return 1;
}

double estimateActionValue(int state, int action, int cn, double alpha, double beta, int statesdivision, int bmax){
int bn;

    bn=state-(bmax+1)*(cn-1);

    inputLayerNeurons[0]=bn;
    inputLayerNeurons[1]=cn;
    //inputLayerNeurons[2]=alpha;
    //inputLayerNeurons[3]=beta;

    ForwardPropagate();

    return outputLayerNeurons[action];
 }

double estimateActionProbability(int state, int action, int cn, double alpha, double beta, int statesdivision, int bmax, int lower, int upper){
int bn;

    bn=state-(bmax+1)*(cn-1);

    inputLayerNeurons[0]=bn;
    inputLayerNeurons[1]=cn;
    //inputLayerNeurons[2]=alpha;
    //inputLayerNeurons[3]=beta;

    ForwardPropagate();
        for (int i=0;i<numOutputNeurons;i++)
            preferences[i]=outputLayerNeurons[i];

    softMaxProbabilities(preferences,lengthActionset,state,statesdivision,lower,upper);   // preferences are now probability values
    printf("preferences[%d]=%lf",action,preferences[action]);

    return preferences[action];
 }

double* mapWeights(double *weightsvector){

    for (int i=0;i<55;i++)
        printf("weightsvector[%d]=%lf",i,weightsvector[i]);

     for (int i = 0; i < numInputNeurons; i++)
		for (int j = 0; j < numHiddenNeurons; j++)
            ihWeights[i][j]=weightsvector[i*numHiddenNeurons+j];

    for (int i = 0; i < numHiddenNeurons; i++)
		for (int j = 0; j < numOutputNeurons; j++)
            hoWeights[i][j]=weightsvector[numInputNeurons*numHiddenNeurons+i*numOutputNeurons+j];

    for (int i = 0; i < numInputNeurons; i++)
		for (int j = 0; j < numHiddenNeurons; j++)
            printf("ihWeights[%d][%d]=%lf",i,j,ihWeights[i][j]);

    for (int i = 0; i < numHiddenNeurons; i++)
		for (int j = 0; j < numOutputNeurons; j++)
            printf("hoWeights[%d][%d]=%lf",i,j,hoWeights[i][j]);// OK

    return weightsvector;
}


/*ANN*/

double sigmoid(double x){
double exp_value;
double return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}

double sigmoidDerivative(double input){
double return_value;

     return_value = sigmoid(input)*(1.0-sigmoid(input));

     return return_value;
}

double ReLuFunction(double input){
    if (input<0)
        return 0;
    else
        return input;
}

double ReLuFunctionDerivative(double input){
    if (input<0)
        return 0;
    else
        return 1;
}

void computeIHLayerOneWeightedSum(){

    for (int iHiddenNeuron=0; iHiddenNeuron<numHiddenNeurons; iHiddenNeuron++){
        ihSums[iHiddenNeuron]=0;
        for (int iInputNeuron=0; iInputNeuron<numInputNeurons; iInputNeuron++){
            ihSums[iHiddenNeuron]=ihSums[iHiddenNeuron]+ihWeights[iInputNeuron][iHiddenNeuron]*inputLayerNeurons[iInputNeuron];
            printf("\n ihSums[%d]=%lf",iHiddenNeuron,ihSums[iHiddenNeuron]);
        }
    }
}

void addBiasesIH(){
    for (int iHiddenNeuron=0; iHiddenNeuron<numHiddenNeurons; iHiddenNeuron++){
        ihSums[iHiddenNeuron]=ihSums[iHiddenNeuron]+ihBiases[iHiddenNeuron];
        printf("\n ihSums[%d]=%lf",iHiddenNeuron,ihSums[iHiddenNeuron]);
    }

}

void computeIHActivation(){
    for (int iHiddenNeuron=0; iHiddenNeuron<numHiddenNeurons; iHiddenNeuron++){
        ihOutputs[iHiddenNeuron]=ReLuFunction(ihSums[iHiddenNeuron]);
        printf("\n ihOutputs[%d]=%lf",iHiddenNeuron,ihOutputs[iHiddenNeuron]);
    }

}

void computeHOWeightedSum(){
double sum=0;
    for (int iOutputNeuron=0; iOutputNeuron<numOutputNeurons; iOutputNeuron++){
        sum=0;
        for (int iHiddenNeuron=0; iHiddenNeuron<numHiddenNeurons; iHiddenNeuron++){
            sum=sum+hoWeights[iHiddenNeuron][iOutputNeuron]*ihOutputs[iHiddenNeuron];
        }
            hoSums[iOutputNeuron]=sum;
            printf("\n hoSums[%d]=%lf",iOutputNeuron,hoSums[iOutputNeuron]);
    }
}

void addBiasesHO(){
    for (int iOutputNeuron=0; iOutputNeuron<numOutputNeurons; iOutputNeuron++){
        hoSums[iOutputNeuron]=hoSums[iOutputNeuron]+hoBiases[iOutputNeuron];
        printf("\n hoSums[%d]=%lf",iOutputNeuron,hoSums[iOutputNeuron]);
    }
}

void computeHOActivation(){
    for (int iOutputNeuron=0; iOutputNeuron<numOutputNeurons; iOutputNeuron++){
        outputLayerNeurons[iOutputNeuron]=ReLuFunction(hoSums[iOutputNeuron]);
        printf("\n outputLayerNeurons[%d]=%lf",iOutputNeuron,outputLayerNeurons[iOutputNeuron]);
    }
}

void ForwardPropagate(){
    computeIHLayerOneWeightedSum();
    addBiasesIH();
    computeIHActivation();
    computeHOWeightedSum();
    addBiasesHO();
    computeHOActivation();
}

void BackwardPropagate(){
    computeErrorOutput();
    computedeltaWeightHO();
    computeErrorHidden();
    adjustWeightsIH();
    adjustWeightsHO();
}

void computeErrorOutput(){
    for (int iOutputNeuron=0; iOutputNeuron<numOutputNeurons; iOutputNeuron++){
        errorOutputNeuron[iOutputNeuron]=(expectedOutputNeuron[iOutputNeuron]-outputLayerNeurons[iOutputNeuron])*ReLuFunctionDerivative(outputLayerNeurons[iOutputNeuron]);
        printf("\n errorOutputNeuron[%d]=%lf",iOutputNeuron,errorOutputNeuron[iOutputNeuron]);
    }
}

void computedeltaWeightHO(){
    for (int iOutputNeuron=0; iOutputNeuron<numOutputNeurons; iOutputNeuron++){
        for (int iHiddenNeuron=0; iHiddenNeuron<numHiddenNeurons; iHiddenNeuron++){
            deltaWeight[iHiddenNeuron][1]=errorOutputNeuron[iOutputNeuron]/ihOutputs[iHiddenNeuron];
            printf("\n hoWeights[%d][%d]=%lf",iHiddenNeuron,iOutputNeuron,hoWeights[iHiddenNeuron][iOutputNeuron]);
        }
    }
}

void computeErrorHidden(){
    for (int iOutputNeuron=0; iOutputNeuron<numOutputNeurons; iOutputNeuron++){
        for (int iHiddenNeuron=0; iHiddenNeuron<numHiddenNeurons; iHiddenNeuron++){
            deltaHiddenSum[iHiddenNeuron][0]=errorOutputNeuron[iOutputNeuron]/hoWeights[iHiddenNeuron][iOutputNeuron]*ReLuFunctionDerivative(ihSums[iHiddenNeuron]);
            printf("\n deltaHiddenSum[%d]=%lf",iHiddenNeuron,deltaHiddenSum[iHiddenNeuron][0]);
        }
    }
}

void adjustWeightsIH(){
    for (int iInputNeuron=0; iInputNeuron<numInputNeurons; iInputNeuron++){
        for (int iHiddenNeuron=0; iHiddenNeuron<numHiddenNeurons; iHiddenNeuron++){
            ihWeights[iInputNeuron][iHiddenNeuron]=ihWeights[iInputNeuron][iHiddenNeuron]+deltaHiddenSum[iHiddenNeuron][0]/inputLayerNeurons[iInputNeuron];
            printf("\n ihWeights[%d][%d]=%lf",iInputNeuron,iHiddenNeuron,ihWeights[iInputNeuron][iHiddenNeuron]);
        }
    }
}

void adjustWeightsHO(){
    for (int iOutputNeuron=0; iOutputNeuron<numOutputNeurons; iOutputNeuron++){
        for (int iHiddenNeuron=0; iHiddenNeuron<numHiddenNeurons; iHiddenNeuron++){
            hoWeights[iHiddenNeuron][iOutputNeuron]=hoWeights[iHiddenNeuron][iOutputNeuron]+deltaWeight[iHiddenNeuron][1];
            printf("\n hoWeights[%d][%d]=%lf",iHiddenNeuron,iOutputNeuron,hoWeights[iHiddenNeuron][iOutputNeuron]);
        }
    }
}

void TrainNetwork(int numItNNTraining){
    for (int itrain=0;itrain<numItNNTraining;itrain++){
        ForwardPropagate();
        BackwardPropagate();
        printf("\ntraining %d complete",itrain);
    }

printf("NN Output=%lf",outputLayerNeurons[0]);
}

/*ANN*/


int simulatePolicy(int batterymax,double alpha,double beta){
int currentState,nextState,currentAction,cn=4,nextcn,bn,packetsize,previouspacketsize,consumption;
int bmax=batterymax;
double batteryratio;
double rndnumber;

for (int runtime=0;runtime<runtimes;runtime++){

        nextcn=gsl_rng_uniform_int(r,3);
        nextcn++;
        bn=gsl_rng_uniform_int(r,bmax);
        nextState=(bmax+1)*(nextcn-1)+bn;//(bmax+1)*nextcn-1;


        for(int epoch=0;epoch<epochs;epoch++){

            currentState=nextState;
            cn=nextcn;
            bn=currentState-(cn-1)*(bmax+1);
            previouspacketsize=dnruntime[muhn-minenergy][epoch-1];

            if (NAIVEPOLICY==0)
                currentAction=epsilonmapping[currentState][0];
            else if (NAIVEPOLICY==1)
                nextcn=FIXEDSECLEVEL;
            else if (NAIVEPOLICY==2){
                if (epoch==0)
                    nextcn=4;
                else if (bn+hnruntime[muhn-minenergy][epoch-1]-eprime[previouspacketsize][3]>0)
                    nextcn=4;
                else if (bn+hnruntime[muhn-minenergy][epoch-1]-eprime[previouspacketsize][2]>0)
                    nextcn=3;
                else if (bn+hnruntime[muhn-minenergy][epoch-1]-eprime[previouspacketsize][1]>0)
                    nextcn=2;
                else
                    nextcn=1;
            }
            else if (NAIVEPOLICY==3){
                if (epoch==0)
                    nextcn=4;
                else if (bn+hnruntime[muhn-minenergy][epoch-1]-eprime[previouspacketsize][3]>0)
                    nextcn=4;
                else if (bn+hnruntime[muhn-minenergy][epoch-1]-eprime[previouspacketsize][2]>0){
                    rndnumber=gsl_rng_uniform(r);
                        if (rndnumber<alpha)
                            nextcn=3;
                        else
                            nextcn=2;
                }
                else if (bn+hnruntime[muhn-minenergy][epoch-1]-eprime[previouspacketsize][1]>0)
                    nextcn=2;
                else
                    nextcn=1;
            }

            securityf[0]=0;
            securityf[1]=1-alpha;
            securityf[2]=alpha;
            securityf[3]=1;

            if (NAIVEPOLICY==0)
                nextcn=cn+currentAction-3;

            if (dnruntime[muhn-minenergy][epoch]<=0)
                packetsize=0;
            else if (dnruntime[muhn-minenergy][epoch]>=80)
                packetsize=80;
            else
                packetsize=dnruntime[muhn-minenergy][epoch];

            consumption=eprime[packetsize][cn-1];
            bn=bn+hnruntime[muhn-minenergy][epoch]-consumption;
            int a=hnruntime[muhn-minenergy][epoch];

            if (bn>bmax){
                bn=bmax;
                batteryratio=bn*1.0/bmax;
                rewardsevosimulation[epoch]=rewardsevosimulation[epoch]+(1-beta)*securityf[nextcn-1]+beta*(batteryratio); //observe R
                //printf("\n rewardsevosimulation[%d]=%lf",epoch,rewardsevosimulation[epoch]);
            }
            else if (bn<=0){
                bn=bn-hnruntime[muhn-minenergy][epoch]+consumption;
                rewardsevosimulation[epoch]=rewardsevosimulation[epoch]+0; //observe R
                discpacketsevosimulation[epoch]=discpacketsevosimulation[epoch]+1;
                //printf("\n rewardsevosimulation[%d]=%lf",epoch,rewardsevosimulation[epoch]);
            }
            else{
                batteryratio=bn*1.0/bmax;
                rewardsevosimulation[epoch]=rewardsevosimulation[epoch]+(1-beta)*securityf[nextcn-1]+beta*(batteryratio); //observe R
                //printf("\n rewardsevosimulation[%d]=%lf",epoch,rewardsevosimulation[epoch]);
            }

            if (NAIVEPOLICY==0)
                nextcn=cn+currentAction-3;

            nextState=(bmax+1)*(nextcn-1)+bn;

        }//for epoch ends
}//for runtime ends
/*--------------------------------------*/
/*------------data treatment------------*/
/*--------------------------------------*/

avgrewardsimulation=0;
avgdiscpacketssimulation=0;
for (int epoch=0;epoch<epochs;epoch++){
    rewardevoovertimesimulation[epoch]=0;
    avgrewardevoovertimesimulation[epoch]=0;
}

// reward
    for(int epoch=0;epoch<epochs;epoch++){
        rewardevoovertimesimulation[epoch]=rewardevoovertimesimulation[epoch]+rewardsevosimulation[epoch];
        avgrewardevoovertimesimulation[epoch]=avgrewardevoovertimesimulation[epoch]+discpacketsevosimulation[epoch];
    }

  /*  for(int epoch=0;epoch<epochs;epoch++){
        rewardevoovertimesimulation[epoch]=rewardevoovertimesimulation[epoch]/(runtimes+1);
    }*/

    for(int epoch=0;epoch<epochs;epoch++){
        avgrewardsimulation=avgrewardsimulation+rewardevoovertimesimulation[epoch];
        avgdiscpacketssimulation=avgdiscpacketssimulation+avgrewardevoovertimesimulation[epoch];
    }

    avgdiscpacketssimulation=avgdiscpacketssimulation/epochs/runtimes;
    avgrewardsimulation=avgrewardsimulation/epochs/runtimes;


/*--------------------------------------*/
/*------------data treatment------------*/
/*--------------------------------------*/

    PrintAvgRewardSimulationVSMuHn(avgrewardsimulation);
    PrintAvgDiscPacketsSimulationVSMuHn(avgdiscpacketssimulation);

    for (int epoch=0;epoch<epochs;epoch++){
        rewardsevosimulation[epoch]=0;
        rewardevoovertimesimulation[epoch]=0;
        avgrewardevoovertimesimulation[epoch]=0;
        discpacketsevosimulation[epoch]=0;
    }

return 1;
}

void PrintAvgRewardSimulationVSMuHn(double avgrewardsimulation){
FILE *AvgRewardSimulationVSMuHnfiledata;
AvgRewardSimulationVSMuHnfiledata=fopen("C:\\Codeblocks\\FrameWorkAlgos\\runtimeresults\\AvgRewardSimulationVSMuHnfiledata.txt","a");

/*    for(int epoch=0;epoch<epochs;epoch=epoch+1)
        fprintf(AvgRewardSimulationVSMuHnfiledata,"%lf ",rewardevoovertimesimulation[epoch]);

    fprintf(AvgRewardSimulationVSMuHnfiledata,"\n");*/
    fprintf(AvgRewardSimulationVSMuHnfiledata,"%lf %d\n",avgrewardsimulation,muhn);

    fclose(AvgRewardSimulationVSMuHnfiledata);

}

void PrintAvgDiscPacketsSimulationVSMuHn(double avgdiscpacketssimulation){
FILE *AvgDiscPacketsSimulationVSMuHnfiledata;
AvgDiscPacketsSimulationVSMuHnfiledata=fopen("C:\\Codeblocks\\FrameWorkAlgos\\runtimeresults\\AvgDiscPacketsSimulationVSMuHnfiledata.txt","a");

    fprintf(AvgDiscPacketsSimulationVSMuHnfiledata,"%lf %d\n",avgdiscpacketssimulation,muhn);

    fclose(AvgDiscPacketsSimulationVSMuHnfiledata);

}

void PrintSecLevelVSTimeDataFile(double alpha, double beta){
FILE *SecLevelVSTimefiledata;
SecLevelVSTimefiledata=fopen("C:\\Codeblocks\\FrameWorkAlgos\\runtimeresults\\SecLevelVSTimefiledata.txt","a");

    fprintf(SecLevelVSTimefiledata,"\n\n");

    for (int epoch=0;epoch<epochs;epoch++){
        fprintf(SecLevelVSTimefiledata,"%lf,",k1[epoch]);
    }
    fprintf(SecLevelVSTimefiledata,"\n\n");

    for (int epoch=0;epoch<epochs;epoch++){
        fprintf(SecLevelVSTimefiledata,"%lf ",k2[epoch]);
    }
    fprintf(SecLevelVSTimefiledata,"\n\n");

    for (int epoch=0;epoch<epochs;epoch++){
        fprintf(SecLevelVSTimefiledata,"%lf ",k3[epoch]);
    }
    fprintf(SecLevelVSTimefiledata,"\n\n");

    for (int epoch=0;epoch<epochs;epoch++){
        fprintf(SecLevelVSTimefiledata,"%lf ",k4[epoch]);
    }
    fprintf(SecLevelVSTimefiledata,"\n");
    fprintf(SecLevelVSTimefiledata,"%lf %lf",alpha,beta);
    fprintf(SecLevelVSTimefiledata,"\n");

fprintf(SecLevelVSTimefiledata,"\n");

fclose(SecLevelVSTimefiledata);
}

void PrintDiscPacketsVSTimeDataFile(double alpha, double beta){
FILE *DiscPacketsVSTimefiledata;
DiscPacketsVSTimefiledata=fopen("C:\\Codeblocks\\FrameWorkAlgos\\runtimeresults\\DiscPacketsVSTimefiledata.txt","a");

    fprintf(DiscPacketsVSTimefiledata,"\n\n");

    for(int epoch=0;epoch<epochs;epoch=epoch+1)
        fprintf(DiscPacketsVSTimefiledata,"%lf ",discardedpacketsovertime[epoch]);

    fprintf(DiscPacketsVSTimefiledata,"\n");
    fprintf(DiscPacketsVSTimefiledata,"%lf %lf",alpha,beta);
    fprintf(DiscPacketsVSTimefiledata,"\n\n");
    fprintf(DiscPacketsVSTimefiledata,"%lf",avgdiscardedpackets);
    fprintf(DiscPacketsVSTimefiledata,"\n");

fclose(DiscPacketsVSTimefiledata);
}

void PrintBatteryEvoVSTimeDataFile(double alpha, double beta){
FILE *BatteryEvoVSTimefiledata;
BatteryEvoVSTimefiledata=fopen("C:\\Codeblocks\\FrameWorkAlgos\\runtimeresults\\BatteryEvoVSTimefiledata.txt","a");

    for(int epoch=0;epoch<epochs;epoch=epoch+1)
        fprintf(BatteryEvoVSTimefiledata,"%lf ",batteryevoovertime[epoch]);

    fprintf(BatteryEvoVSTimefiledata,"\n");
    fprintf(BatteryEvoVSTimefiledata,"%lf %lf",alpha,beta);
    fprintf(BatteryEvoVSTimefiledata,"\n");

fclose(BatteryEvoVSTimefiledata);
}

void PrintRewardEvoVSTimeDataFile(double alpha, double beta){
FILE *RewardEvoVSTimefiledata;
RewardEvoVSTimefiledata=fopen("C:\\Codeblocks\\FrameWorkAlgos\\runtimeresults\\RewardEvoVSTimefiledata.txt","a");

    for(int epoch=0;epoch<epochs;epoch=epoch+1)
        fprintf(RewardEvoVSTimefiledata,"%lf ",rewardevoovertime[epoch]);

    fprintf(RewardEvoVSTimefiledata,"\n");
    fprintf(RewardEvoVSTimefiledata,"%lf %lf",alpha,beta);
    fprintf(RewardEvoVSTimefiledata,"\n");

fclose(RewardEvoVSTimefiledata);
}

void readHarvesterAndPacketdata(){
FILE *HarvesterAndPacketfiledata;
HarvesterAndPacketfiledata=fopen("C:\\Codeblocks\\FrameWorkAlgos\\HarvesterAndPacketfiledatasigmadn1sigmahn1.txt","r");

for (int muhnaux=minenergy;muhnaux<=maxenergy;muhnaux++){
    for (int iepoch=0;iepoch<=epochs;iepoch++){
        fscanf(HarvesterAndPacketfiledata,"%d",&hnruntime[muhnaux-minenergy][iepoch]);
        printf("hnruntime[%d][%d]=%d\n",muhnaux-minenergy,iepoch,hnruntime[muhnaux-minenergy][iepoch]);
    }
}

for (int muhnaux=minenergy;muhnaux<=maxenergy;muhnaux++){
    for (int iepoch=0;iepoch<=epochs;iepoch++){
        fscanf(HarvesterAndPacketfiledata,"%d",&dnruntime[muhnaux-minenergy][iepoch]);
        printf("dnruntime[%d][%d]=%d\n",muhnaux-minenergy,iepoch,dnruntime[muhnaux-minenergy][iepoch]);
    }
}

}
