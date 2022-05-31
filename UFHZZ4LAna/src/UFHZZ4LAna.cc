// -*- C++ -*-
//
// Package:    UFHZZ4LAna
// Class:      UFHZZ4LAna
// 
//

// system include files
#include <memory>
#include <string>
#include <map>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <set>

#define PI 3.14159

// user include files 
#include "TROOT.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TSpline.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TString.h"
#include "TLorentzVector.h"
#include "Math/VectorUtil.h"
#include "TClonesArray.h"
#include "TCanvas.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/MergeableCounter.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//HTXS
#include "SimDataFormats/HTXS/interface/HiggsTemplateCrossSections.h"
//#include "SimDataFormats/HZZFiducial/interface/HZZFiducialVolume.h"

// PAT
#include "DataFormats/PatCandidates/interface/PFParticle.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

// Reco
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

// KD's
//#include "JHUGenMELA/MELA/interface/Mela.h"

//Helper
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LHelper.h"
//Muons
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LMuonAna.h"
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LMuonTree.h"
//Electrons
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LElectronTree.h"
//Photons
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LPhotonTree.h"
//Jets
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LJetTree.h"
//Final Leps
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LFinalLepTree.h"
//Sip
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LSipAna.h"
//PU
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LPileUp.h"
#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

//GEN
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LGENAna.h"
//VBF Jets
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/HZZ4LJets.h"

// Jet energy correction
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include <vector>

// Kinematic Fit
//#include "KinZfitter/KinZfitter/interface/KinZfitter.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

// EWK corrections
#include "UFHZZAnalysisRun2/UFHZZ4LAna/interface/EwkCorrections.h"

// JEC related
#include "PhysicsTools/PatAlgos/plugins/PATJetUpdater.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
 
//JER related
#include "JetMETCorrections/Modules/interface/JetResolution.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

//BTag Calibration

#include "CondFormats/BTauObjects/interface/BTagCalibration.h"
#include "CondTools/BTau/interface/BTagCalibrationReader.h"

//Muon MVA
//#include "MuonMVAReader/Reader/interface/MuonGBRForestReader.hpp"

// KalmanVertexFitter  
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/InvariantMassFromVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
// Rochester Corrections
#include "UFHZZAnalysisRun2/KalmanMuonCalibrationsProducer/src/RoccoR.cc"

#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"


//
// class declaration
//
using namespace EwkCorrections;

class UFHZZ4LAna : public edm::EDAnalyzer {
public:
    explicit UFHZZ4LAna(const edm::ParameterSet&);
    ~UFHZZ4LAna();
  
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    static bool sortByPt( const reco::GenParticle &p1, const reco::GenParticle &p2 ){ return (p1.pt() > p2.pt()); };
  
private:
    virtual void beginJob() ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
  
    virtual void beginRun(edm::Run const&, const edm::EventSetup& iSetup);
    virtual void endRun(edm::Run const&, edm::EventSetup const&);
    virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
    virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,edm::EventSetup const& eSetup);
  
   // void findHiggsCandidate(std::vector< pat::Muon > &selectedMuons, std::vector< pat::Electron > &selectedElectrons, const edm::Event& iEvent, const edm::EventSetup& iSetup);
   // void findZ1LCandidate(const edm::Event& iEvent);
    
   	RoccoR  *calibrator;
	//float ApplyRoccoR(int Y, bool isMC, int charge, float pt, float eta, float phi, float genPt, float nLayers);

    //Helper Class
    HZZ4LHelper helper;
    //GEN
    HZZ4LGENAna genAna;
    //VBF
    HZZ4LJets jetHelper;
    //PU Reweighting
    edm::LumiReWeighting *lumiWeight;
    HZZ4LPileUp pileUp;
    //JES Uncertainties
    std::unique_ptr<JetCorrectionUncertainty> jecunc;
    // kfactors
    TSpline3 *kFactor_ggzz;
    std::vector<std::vector<float> > tableEwk;
    // data/MC scale factors
    TH2F *hElecScaleFac;
    TH2F *hElecScaleFac_Cracks;
    TH2F *hElecScaleFacGsf;
    TH2F *hElecScaleFacGsfLowET;
    TH2F *hMuScaleFac;
    TH2F *hMuScaleFacUnc;
    TH1D *h_pileup;
    TH1D *h_pileupUp;
    TH1D *h_pileupDn;
    std::vector<TH1F*> h_medians;
    TH2F *hbTagEffi;
    TH2F *hcTagEffi;
    TH2F *hudsgTagEffi;

    BTagCalibrationReader* reader;

    //Saved Events Trees
    TTree *passedEventsTree_All;

    void bookPassedEventTree(TString treeName, TTree *tree);
    void setTreeVariables( const edm::Event&, const edm::EventSetup&, 
                           std::vector<pat::Jet> goodJets, std::vector<float> goodJetQGTagger, 
                           std::vector<float> goodJetaxis2, std::vector<float> goodJetptD, std::vector<int> goodJetmult, 
                           std::vector<pat::Jet> selectedMergedJets);
    void setGENVariables(edm::Handle<edm::View<reco::GenJet> > genJets);
    /*void setTreeVariables( const edm::Event&, const edm::EventSetup&, 
                           //std::vector<pat::Muon> selectedMuons, std::vector<pat::Electron> selectedElectrons, 
                           //std::vector<pat::Muon> recoMuons, std::vector<pat::Electron> recoElectrons, 
                           std::vector<pat::Jet> goodJets, std::vector<float> goodJetQGTagger, 
                           std::vector<float> goodJetaxis2, std::vector<float> goodJetptD, std::vector<int> goodJetmult, 
                           std::vector<pat::Jet> selectedMergedJets,
                           //std::map<unsigned int, TLorentzVector> selectedFsrMap);
    void setGENVariables(//edm::Handle<reco::GenParticleCollection> prunedgenParticles,
                         //edm::Handle<edm::View<pat::PackedGenParticle> > packedgenParticles,
                         edm::Handle<edm::View<reco::GenJet> > genJets);*/
    //bool mZ1_mZ2(unsigned int& L1, unsigned int& L2, unsigned int& L3, unsigned int& L4, bool makeCuts);

    // -------------------------
    // RECO level information
    // -------------------------

    // Event Variables
    ULong64_t Run, Event, LumiSect;
    int nVtx, nInt;
    int finalState;
    std::string triggersPassed;
    bool passedTrig, passedFullSelection, passedZ4lSelection, passedQCDcut;
    bool passedZ1LSelection, passedZ4lZ1LSelection, passedZ4lZXCRSelection, passedZXCRSelection;
    int nZXCRFailedLeptons;  
    
    float PV_x, PV_y, PV_z; 
    float BS_x, BS_y, BS_z; 
    float BS_xErr, BS_yErr, BS_zErr; 
    float BeamWidth_x, BeamWidth_y;
    float BeamWidth_xErr, BeamWidth_yErr;


    // Event Weights
    float genWeight, pileupWeight, pileupWeightUp, pileupWeightDn, dataMCWeight, eventWeight, prefiringWeight;
    float k_ggZZ, k_qqZZ_qcd_dPhi, k_qqZZ_qcd_M, k_qqZZ_qcd_Pt, k_qqZZ_ewk;
    // pdf weights                                                                   
    vector<float> qcdWeights;
    vector<float> nnloWeights;
    vector<float> pdfWeights;
    int posNNPDF;
    float pdfRMSup, pdfRMSdown, pdfENVup, pdfENVdown;
    
    // lepton variables
    /*vector<double> lep_pt_FromMuonBestTrack, lep_eta_FromMuonBestTrack, lep_phi_FromMuonBestTrack;
    vector<double> lep_position_x, lep_position_y, lep_position_z;
    vector<double> lep_pt_genFromReco;
    vector<double> lep_pt; vector<double> lep_pterr; vector<double> lep_pterrold; 
    vector<double> lep_p; vector<double> lep_ecalEnergy; vector<int> lep_isEB; vector<int> lep_isEE;
    vector<double> lep_eta; vector<double> lep_phi; vector<double> lep_mass;
    vector<double> lepFSR_pt; vector<double> lepFSR_eta; vector<double> lepFSR_phi; vector<double> lepFSR_mass; vector<int> lepFSR_ID;

    vector<double> lep_errPre_Scale, lep_errPost_Scale, lep_errPre_noScale, lep_errPost_noScale;
    vector<double> lep_pt_UnS, lep_pterrold_UnS;

    int lep_Hindex[4];//position of Higgs candidate leptons in lep_p4: 0 = Z1 lead, 1 = Z1 sub, 2 = Z2 lead, 3 = Z2 sub
    float pTL1, pTL2, pTL3, pTL4;
    float etaL1, etaL2, etaL3, etaL4;
    float phiL1, phiL2, phiL3, phiL4;
    int idL1, idL2, idL3, idL4;
    float mL1, mL2, mL3, mL4;
    float pTErrL1, pTErrL2, pTErrL3, pTErrL4;

    float pTL1FSR, pTL2FSR, pTL3FSR, pTL4FSR;
    float etaL1FSR, etaL2FSR, etaL3FSR, etaL4FSR;
    float phiL1FSR, phiL2FSR, phiL3FSR, phiL4FSR;
    float mL1FSR, mL2FSR, mL3FSR, mL4FSR;
    float pTErrL1FSR, pTErrL2FSR, pTErrL3FSR, pTErrL4FSR;

    vector<float> lep_d0BS;
	vector<float> lep_numberOfValidPixelHits;
	vector<float> lep_trackerLayersWithMeasurement;

    vector<float> lep_d0PV;
    vector<float> lep_dataMC; vector<float> lep_dataMCErr;
    vector<float> dataMC_VxBS; vector<float> dataMCErr_VxBS;
    vector<int> lep_genindex; //position of lepton in GENlep_p4 (if gen matched, -1 if not gen matched)
    vector<int> lep_matchedR03_PdgId, lep_matchedR03_MomId, lep_matchedR03_MomMomId; // gen matching even if not in GENlep_p4
    vector<int> lep_id;
    vector<float> lep_mva; vector<int> lep_ecalDriven; 
    vector<int> lep_tightId; vector<int> lep_tightIdSUS; vector<int> lep_tightIdHiPt; //vector<int> lep_tightId_old;
    vector<float> lep_Sip; vector<float> lep_IP; vector<float> lep_isoNH; vector<float> lep_isoCH; vector<float> lep_isoPhot;
    vector<float> lep_isoPU; vector<float> lep_isoPUcorr; 
    vector<float> lep_RelIso; vector<float> lep_RelIsoNoFSR; vector<float> lep_MiniIso; 
    vector<float> lep_ptRatio; vector<float> lep_ptRel;
    vector<int> lep_missingHits;
    vector<string> lep_filtersMatched; // for each lepton, all filters it is matched to
    int nisoleptons;
    double muRho, elRho, rhoSUS;*/

    // tau variables
    /*vector<int> tau_id;
    vector<double> tau_pt, tau_eta, tau_phi, tau_mass;*/

    // photon variables
    //vector<double> pho_pt, pho_eta, pho_phi, photonCutBasedIDLoose;

    // Higgs candidate variables

	/*vector<TLorentzVector> vtxLep_BS;
	vector<TLorentzVector> vtxLep;
	vector<TLorentzVector> vtxRecoLep_BS;


	vector<double> vtxLep_BS_pt_NoRoch;	vector<double> vtxLep_BS_pt; 	vector<double> vtxLep_BS_ptError; 	vector<double> vtxLep_BS_eta; 	vector<double> vtxLep_BS_phi; 	vector<double> vtxLep_BS_mass; vector<double> vtxLep_BS_d0;
	vector<double> vtxLep_pt; 	vector<double> vtxLep_ptError;	vector<double> vtxLep_eta; 	vector<double> vtxLep_phi; 	vector<double> vtxLep_mass;

	vector<double> vtxLepFSR_BS_pt; 	vector<double> vtxLepFSR_BS_eta; 	vector<double> vtxLepFSR_BS_phi; 	vector<double> vtxLepFSR_BS_mass;
	vector<double> vtxLepFSR_pt; 	vector<double> vtxLepFSR_eta; 	vector<double> vtxLepFSR_phi; 	vector<double> vtxLepFSR_mass;*/


    /*vector<double> H_pt; vector<double> H_eta; vector<double> H_phi; vector<double> H_mass;
    vector<double> H_noFSR_pt; vector<double> H_noFSR_eta; vector<double> H_noFSR_phi; vector<double> H_noFSR_mass;
    float mass4l, mass4l_noFSR, mass4e, mass4mu, mass2e2mu, pT4l, eta4l, phi4l, rapidity4l;
    
    float mass3l;

	float massH_vtx_chi2;
	float massH_vtx_chi2_BS;*/

 

    // Z candidate variables
    /*float mass2l_vtx_BS;
    float mass2l_vtx;
	float massZ_vtx_chi2;
	float massZ_vtx_chi2_BS;


    vector<double> Z_pt; vector<double> Z_eta; vector<double> Z_phi; vector<double> Z_mass;
    vector<double> Z_noFSR_pt; vector<double> Z_noFSR_eta; vector<double> Z_noFSR_phi; vector<double> Z_noFSR_mass;
    int Z_Hindex[2]; // position of Z1 and Z2 in Z_p4
    float massZ1, massZ1_Z1L, massZ2, pTZ1, pTZ2;
    float massErrH_vtx;*/

    // MET
    float met; float met_phi;
    float met_jesup, met_phi_jesup, met_jesdn, met_phi_jesdn;
    float met_uncenup, met_phi_uncenup, met_uncendn, met_phi_uncendn;

    // Jets
    vector<int>    jet_iscleanH4l;
    int jet1index, jet2index;
    vector<double> jet_pt; vector<double> jet_eta; vector<double> jet_phi; vector<double> jet_mass; vector<double> jet_pt_raw;
    vector<float>  jet_pumva, jet_csvv2,  jet_csvv2_; vector<int> jet_isbtag;
    vector<int>    jet_hadronFlavour, jet_partonFlavour;
    vector<float>  jet_QGTagger, jet_QGTagger_jesup, jet_QGTagger_jesdn; 
    vector<float> jet_axis2, jet_ptD; vector<int> jet_mult;
    vector<float>  jet_relpterr; vector<float>  jet_phierr;
    vector<float>  jet_bTagEffi;
    vector<float>  jet_cTagEffi;
    vector<float>  jet_udsgTagEffi;
    vector<int>    jet_jesup_iscleanH4l;
    vector<double> jet_jesup_pt; vector<double> jet_jesup_eta; 
    vector<double> jet_jesup_phi; vector<double> jet_jesup_mass;
    vector<int>    jet_jesdn_iscleanH4l;
    vector<double> jet_jesdn_pt; vector<double> jet_jesdn_eta; 
    vector<double> jet_jesdn_phi; vector<double> jet_jesdn_mass;
    vector<int>    jet_jerup_iscleanH4l;
    vector<double> jet_jerup_pt; vector<double> jet_jerup_eta; 
    vector<double> jet_jerup_phi; vector<double> jet_jerup_mass;
    vector<int>    jet_jerdn_iscleanH4l;
    vector<double> jet_jerdn_pt; vector<double> jet_jerdn_eta; 
    vector<double> jet_jerdn_phi; vector<double> jet_jerdn_mass;    
    int njets_pt30_eta4p7; int njets_pt30_eta4p7_jesup; int njets_pt30_eta4p7_jesdn; 
    int njets_pt30_eta4p7_jerup; int njets_pt30_eta4p7_jerdn;
    int njets_pt30_eta2p5; int njets_pt30_eta2p5_jesup; int njets_pt30_eta2p5_jesdn; 
    int njets_pt30_eta2p5_jerup; int njets_pt30_eta2p5_jerdn;
    int nbjets_pt30_eta4p7; int nvjets_pt40_eta2p4;
    float pt_leadingjet_pt30_eta4p7;
    float pt_leadingjet_pt30_eta4p7_jesup; float pt_leadingjet_pt30_eta4p7_jesdn;
    float pt_leadingjet_pt30_eta4p7_jerup; float pt_leadingjet_pt30_eta4p7_jerdn;
    float pt_leadingjet_pt30_eta2p5;
    float pt_leadingjet_pt30_eta2p5_jesup; float pt_leadingjet_pt30_eta2p5_jesdn;
    float pt_leadingjet_pt30_eta2p5_jerup; float pt_leadingjet_pt30_eta2p5_jerdn;
    float absrapidity_leadingjet_pt30_eta4p7;
    float absrapidity_leadingjet_pt30_eta4p7_jesup; float absrapidity_leadingjet_pt30_eta4p7_jesdn;
    float absrapidity_leadingjet_pt30_eta4p7_jerup; float absrapidity_leadingjet_pt30_eta4p7_jerdn;
    float absdeltarapidity_hleadingjet_pt30_eta4p7;
    float absdeltarapidity_hleadingjet_pt30_eta4p7_jesup; float absdeltarapidity_hleadingjet_pt30_eta4p7_jesdn;
    float absdeltarapidity_hleadingjet_pt30_eta4p7_jerup; float absdeltarapidity_hleadingjet_pt30_eta4p7_jerdn;
    float DijetMass, DijetDEta, DijetFisher;

    // merged jets
    vector<int>   mergedjet_iscleanH4l;
    vector<float> mergedjet_pt; vector<float> mergedjet_eta; vector<float> mergedjet_phi; vector<float> mergedjet_mass;
    
    vector<float> mergedjet_tau1; vector<float> mergedjet_tau2;
    vector<float> mergedjet_btag;

    vector<float> mergedjet_L1;
    vector<float> mergedjet_prunedmass; vector<float> mergedjet_softdropmass;
    
    vector<int> mergedjet_nsubjet;
    vector<vector<float> > mergedjet_subjet_pt; vector<vector<float> > mergedjet_subjet_eta;
    vector<vector<float> > mergedjet_subjet_phi; vector<vector<float> > mergedjet_subjet_mass;
    vector<vector<float> > mergedjet_subjet_btag;
    vector<vector<int> > mergedjet_subjet_partonFlavour, mergedjet_subjet_hadronFlavour;

    // FSR Photons
    /*int nFSRPhotons;
    vector<int> fsrPhotons_lepindex; 
    vector<double> fsrPhotons_pt; vector<double> fsrPhotons_pterr;
    vector<double> fsrPhotons_eta; vector<double> fsrPhotons_phi;
    vector<double> fsrPhotons_mass;
    vector<float> fsrPhotons_dR; vector<float> fsrPhotons_iso;
    vector<float> allfsrPhotons_dR; vector<float> allfsrPhotons_pt; vector<float> allfsrPhotons_iso;*/

    // Z4l? FIXME
    /*/float theta12, theta13, theta14;  
    float minM3l, Z4lmaxP, minDeltR, m3l_soft;
    float minMass2Lep, maxMass2Lep;
    float thetaPhoton, thetaPhotonZ;*/

    // Event Category
    int EventCat;

    // -------------------------
    // GEN level information
    // -------------------------

    //Event variables
    int GENfinalState;
    bool passedFiducialSelection;

    // lepton variables
    /*vector<double> GENlep_pt; vector<double> GENlep_eta; vector<double> GENlep_phi; vector<double> GENlep_mass; 
    vector<int> GENlep_id; vector<int> GENlep_status; 
    vector<int> GENlep_MomId; vector<int> GENlep_MomMomId;
    int GENlep_Hindex[4];//position of Higgs candidate leptons in lep_p4: 0 = Z1 lead, 1 = Z1 sub, 2 = Z2 lead, 3 = Z3 sub
    vector<float> GENlep_isoCH; vector<float> GENlep_isoNH; vector<float> GENlep_isoPhot; vector<float> GENlep_RelIso; 

    // Higgs candidate variables (calculated using selected gen leptons)
    vector<double> GENH_pt; vector<double> GENH_eta; vector<double> GENH_phi; vector<double> GENH_mass; 
    float GENmass4l, GENmass4e, GENmass4mu, GENmass2e2mu, GENpT4l, GENeta4l, GENrapidity4l;
    float GENMH; //mass directly from gen particle with id==25
    

    // Z candidate variables
    vector<double> GENZ_pt; vector<double> GENZ_eta; vector<double> GENZ_phi; vector<double> GENZ_mass; 
    vector<int> GENZ_DaughtersId; vector<int> GENZ_MomId;
    float  GENmassZ1, GENmassZ2, GENpTZ1, GENpTZ2, GENdPhiZZ, GENmassZZ, GENpTZZ;

    // Higgs variables directly from GEN particle
    float GENHmass;*/

    // Jets
    vector<double> GENjet_pt; vector<double> GENjet_eta; vector<double> GENjet_phi; vector<double> GENjet_mass; 
    int GENnjets_pt30_eta4p7; float GENpt_leadingjet_pt30_eta4p7; 
    int GENnjets_pt30_eta2p5; float GENpt_leadingjet_pt30_eta2p5; 
    float GENabsrapidity_leadingjet_pt30_eta4p7; float GENabsdeltarapidity_hleadingjet_pt30_eta4p7;
    int lheNb, lheNj, nGenStatus2bHad;

    

    // MEM


    // a vector<float> for each vector<double>
   /* vector<float> lep_d0BS_float;
    vector<float> lep_d0PV_float;

	vector<float> lep_numberOfValidPixelHits_float;
	vector<float> lep_trackerLayersWithMeasurement_float;


	vector<double> vtxLep_BS_pt_NoRoch_float; 		vector<double> vtxLep_BS_pt_float; 	vector<double> vtxLep_BS_ptError_float; 	vector<double> vtxLep_BS_eta_float; 	vector<double> vtxLep_BS_phi_float; 	vector<double> vtxLep_BS_mass_float; vector<double> vtxLep_BS_d0_float;
	vector<double> vtxLep_pt_float; 	vector<double> vtxLep_ptError_float; 	vector<double> vtxLep_eta_float; 	vector<double> vtxLep_phi_float; 	vector<double> vtxLep_mass_float;
	
	vector<double> vtxLepFSR_BS_pt_float; 	vector<double> vtxLepFSR_BS_eta_float; 	vector<double> vtxLepFSR_BS_phi_float; 	vector<double> vtxLepFSR_BS_mass_float;
	vector<double> vtxLepFSR_pt_float; 	vector<double> vtxLepFSR_eta_float; 	vector<double> vtxLepFSR_phi_float; 	vector<double> vtxLepFSR_mass_float;*/
	
	




	/*vector<float> lep_pt_FromMuonBestTrack_float, lep_eta_FromMuonBestTrack_float, lep_phi_FromMuonBestTrack_float;
	vector<float> lep_position_x_float, lep_position_y_float, lep_position_z_float;
	vector<float> lep_pt_genFromReco_float;
    vector<double> lep_pt_UnS_float, lep_pterrold_UnS_float;
    vector<float> lep_errPre_Scale_float;
    vector<float> lep_errPost_Scale_float;
    vector<float> lep_errPre_noScale_float;
    vector<float> lep_errPost_noScale_float;

    vector<float> lep_pt_float, lep_pterr_float, lep_pterrold_float;
    vector<float> lep_p_float, lep_ecalEnergy_float;
    vector<float> lep_eta_float, lep_phi_float, lep_mass_float;
    vector<float> lepFSR_pt_float, lepFSR_eta_float;
    vector<float> lepFSR_phi_float, lepFSR_mass_float;
    vector<float> tau_pt_float, tau_eta_float, tau_phi_float, tau_mass_float;
    vector<float> pho_pt_float, pho_eta_float, pho_phi_float, photonCutBasedIDLoose_float;
    vector<float> H_pt_float, H_eta_float, H_phi_float, H_mass_float;
    vector<float> H_noFSR_pt_float, H_noFSR_eta_float; 
    vector<float> H_noFSR_phi_float, H_noFSR_mass_float;
    vector<float> Z_pt_float, Z_eta_float, Z_phi_float, Z_mass_float;
    vector<float> Z_noFSR_pt_float, Z_noFSR_eta_float;
    vector<float> Z_noFSR_phi_float, Z_noFSR_mass_float;*/
    vector<float> jet_pt_float, jet_eta_float, jet_phi_float, jet_mass_float, jet_pt_raw_float;
    vector<float> jet_jesup_pt_float, jet_jesup_eta_float; 
    vector<float> jet_jesup_phi_float, jet_jesup_mass_float;
    vector<float> jet_jesdn_pt_float, jet_jesdn_eta_float;
    vector<float> jet_jesdn_phi_float, jet_jesdn_mass_float;
    vector<float> jet_jerup_pt_float, jet_jerup_eta_float;
    vector<float> jet_jerup_phi_float, jet_jerup_mass_float;
    vector<float> jet_jerdn_pt_float, jet_jerdn_eta_float;
    vector<float> jet_jerdn_phi_float, jet_jerdn_mass_float;
    vector<float> fsrPhotons_pt_float, fsrPhotons_pterr_float;
    vector<float> fsrPhotons_eta_float, fsrPhotons_phi_float, fsrPhotons_mass_float;
    vector<float> GENlep_pt_float, GENlep_eta_float;
    vector<float> GENlep_phi_float, GENlep_mass_float;
    vector<float> GENH_pt_float, GENH_eta_float;
    vector<float> GENH_phi_float, GENH_mass_float;
    vector<float> GENZ_pt_float, GENZ_eta_float;
    vector<float> GENZ_phi_float, GENZ_mass_float;
    vector<float> GENjet_pt_float, GENjet_eta_float;
    vector<float> GENjet_phi_float, GENjet_mass_float;

    // Global Variables but not stored in the tree
   /* vector<double> lep_ptreco;
    vector<int> lep_ptid; vector<int> lep_ptindex;
    vector<pat::Muon> recoMuons; vector<pat::Electron> recoElectrons; vector<pat::Electron> recoElectronsUnS; 
    vector<pat::Tau> recoTaus; vector<pat::Photon> recoPhotons;
    vector<pat::PFParticle> fsrPhotons; 
    TLorentzVector HVec, HVecNoFSR, Z1Vec, Z2Vec;
    TLorentzVector GENZ1Vec, GENZ2Vec;
    bool RecoFourMuEvent, RecoFourEEvent, RecoTwoETwoMuEvent, RecoTwoMuTwoEEvent;
    bool foundHiggsCandidate; bool foundZ1LCandidate; bool firstEntry;*/
    float jet1pt, jet2pt;
    bool firstEntry;

    // hist container
    std::map<std::string,TH1F*> histContainer_;

    //Input edm
    /*edm::EDGetTokenT<edm::View<pat::Electron> > elecSrc_;
    edm::EDGetTokenT<edm::View<pat::Electron> > elecUnSSrc_;
    edm::EDGetTokenT<edm::View<pat::Muon> > muonSrc_;
    edm::EDGetTokenT<edm::View<pat::Tau> > tauSrc_;
    edm::EDGetTokenT<edm::View<pat::Photon> > photonSrc_;*/
    edm::EDGetTokenT<edm::View<pat::Jet> > jetSrc_;
    edm::EDGetTokenT<edm::ValueMap<float> > qgTagSrc_;
    edm::EDGetTokenT<edm::ValueMap<float> > axis2Src_;
    edm::EDGetTokenT<edm::ValueMap<int> > multSrc_;
    edm::EDGetTokenT<edm::ValueMap<float> > ptDSrc_;
    edm::EDGetTokenT<edm::View<pat::Jet> > mergedjetSrc_;
    edm::EDGetTokenT<edm::View<pat::MET> > metSrc_;
    //edm::InputTag triggerSrc_;
    edm::EDGetTokenT<edm::TriggerResults> triggerSrc_;
    edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> triggerObjects_;
    edm::EDGetTokenT<reco::VertexCollection> vertexSrc_;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotSrc_;
    edm::EDGetTokenT<std::vector<reco::Conversion> > conversionSrc_;
    edm::EDGetTokenT<double> muRhoSrc_;
    edm::EDGetTokenT<double> elRhoSrc_;
    edm::EDGetTokenT<double> rhoSrcSUS_;
    edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileupSrc_;
    edm::EDGetTokenT<pat::PackedCandidateCollection> pfCandsSrc_;
    edm::EDGetTokenT<edm::View<pat::PFParticle> > fsrPhotonsSrc_;
    edm::EDGetTokenT<reco::GenParticleCollection> prunedgenParticlesSrc_;
    edm::EDGetTokenT<edm::View<pat::PackedGenParticle> > packedgenParticlesSrc_;
    edm::EDGetTokenT<edm::View<reco::GenJet> > genJetsSrc_;
    edm::EDGetTokenT<GenEventInfoProduct> generatorSrc_;
    edm::EDGetTokenT<LHEEventProduct> lheInfoSrc_;
    edm::EDGetTokenT<LHERunInfoProduct> lheRunInfoToken_;
    edm::EDGetTokenT<HTXS::HiggsClassification> htxsSrc_;
    //edm::EDGetTokenT<HZZFid::FiducialSummary> fidRivetSrc_;
    edm::EDGetTokenT< double > prefweight_token_;

    // Configuration
    const float Zmass;
    float mZ1Low, mZ2Low, mZ1High, mZ2High, m4lLowCut;
    float jetpt_cut, jeteta_cut;
    std::string elecID;
    bool isMC, isSignal;
    float mH;
    float crossSection;
    bool weightEvents;
    float isoCutEl, isoCutMu; 
    double isoConeSizeEl, isoConeSizeMu;
    float sip3dCut, leadingPtCut, subleadingPtCut;
    float genIsoCutEl, genIsoCutMu;
    double genIsoConeSizeEl, genIsoConeSizeMu;
    float _elecPtCut, _muPtCut, _tauPtCut, _phoPtCut;
    float BTagCut;
    bool reweightForPU;
    std::string PUVersion;
    bool doFsrRecovery,bestCandMela, doMela, GENbestM4l;
    bool doPUJetID;
    int jetIDLevel;
    bool doJER;
    bool doJEC;
    bool doRefit;
    bool doTriggerMatching;
    bool checkOnlySingle;
    std::vector<std::string> triggerList;
    int skimLooseLeptons, skimTightLeptons;
    bool verbose;

    int year;///use to choose Muon BDT
    bool isCode4l;

    // register to the TFileService
    edm::Service<TFileService> fs;

    // Counters
    float nEventsTotal;

    float sumWeightsTotal;
    float sumWeightsTotalPU;

    // JER
    JME::JetResolution resolution_pt, resolution_phi;
    JME::JetResolutionScaleFactor resolution_sf;

    string EleBDT_name_161718;
    string heepID_name_161718;

};  //end class declaration


UFHZZ4LAna::UFHZZ4LAna(const edm::ParameterSet& iConfig) :
	histContainer_(),
	jetSrc_(consumes<edm::View<pat::Jet> >(iConfig.getUntrackedParameter<edm::InputTag>("jetSrc"))),  
  qgTagSrc_(consumes<edm::ValueMap<float>>(edm::InputTag("QGTagger", "qgLikelihood"))),
  axis2Src_(consumes<edm::ValueMap<float>>(edm::InputTag("QGTagger", "axis2"))),
  multSrc_(consumes<edm::ValueMap<int>>(edm::InputTag("QGTagger", "mult"))),
  ptDSrc_(consumes<edm::ValueMap<float>>(edm::InputTag("QGTagger", "ptD"))),
  mergedjetSrc_(consumes<edm::View<pat::Jet> >(iConfig.getUntrackedParameter<edm::InputTag>("mergedjetSrc"))),
  metSrc_(consumes<edm::View<pat::MET> >(iConfig.getUntrackedParameter<edm::InputTag>("metSrc"))),
  triggerSrc_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("triggerSrc"))),
  triggerObjects_(consumes<pat::TriggerObjectStandAloneCollection>(iConfig.getParameter<edm::InputTag>("triggerObjects"))),
  vertexSrc_(consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vertexSrc"))),
  beamSpotSrc_(consumes<reco::BeamSpot>(iConfig.getUntrackedParameter<edm::InputTag>("beamSpotSrc"))),
  conversionSrc_(consumes<std::vector<reco::Conversion> >(iConfig.getUntrackedParameter<edm::InputTag>("conversionSrc"))),
  muRhoSrc_(consumes<double>(iConfig.getUntrackedParameter<edm::InputTag>("muRhoSrc"))),
  elRhoSrc_(consumes<double>(iConfig.getUntrackedParameter<edm::InputTag>("elRhoSrc"))),
  rhoSrcSUS_(consumes<double>(iConfig.getUntrackedParameter<edm::InputTag>("rhoSrcSUS"))),
  pileupSrc_(consumes<std::vector<PileupSummaryInfo> >(iConfig.getUntrackedParameter<edm::InputTag>("pileupSrc"))),
  pfCandsSrc_(consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("pfCandsSrc"))),
  fsrPhotonsSrc_(consumes<edm::View<pat::PFParticle> >(iConfig.getUntrackedParameter<edm::InputTag>("fsrPhotonsSrc"))),
  prunedgenParticlesSrc_(consumes<reco::GenParticleCollection>(iConfig.getUntrackedParameter<edm::InputTag>("prunedgenParticlesSrc"))),
  packedgenParticlesSrc_(consumes<edm::View<pat::PackedGenParticle> >(iConfig.getUntrackedParameter<edm::InputTag>("packedgenParticlesSrc"))),
  genJetsSrc_(consumes<edm::View<reco::GenJet> >(iConfig.getUntrackedParameter<edm::InputTag>("genJetsSrc"))),
  generatorSrc_(consumes<GenEventInfoProduct>(iConfig.getUntrackedParameter<edm::InputTag>("generatorSrc"))),
  lheInfoSrc_(consumes<LHEEventProduct>(iConfig.getUntrackedParameter<edm::InputTag>("lheInfoSrc"))),
  lheRunInfoToken_(consumes<LHERunInfoProduct,edm::InRun>(edm::InputTag("externalLHEProducer",""))),
  htxsSrc_(consumes<HTXS::HiggsClassification>(edm::InputTag("rivetProducerHTXS","HiggsClassification"))),
  prefweight_token_(consumes< double >(edm::InputTag("prefiringweight:nonPrefiringProb"))),
	Zmass(91.1876),
  mZ1Low(iConfig.getUntrackedParameter<double>("mZ1Low",40.0)),
  mZ2Low(iConfig.getUntrackedParameter<double>("mZ2Low",12.0)), // was 12
  mZ1High(iConfig.getUntrackedParameter<double>("mZ1High",120.0)),
  mZ2High(iConfig.getUntrackedParameter<double>("mZ2High",120.0)),
  m4lLowCut(iConfig.getUntrackedParameter<double>("m4lLowCut",70.0)),
//     m4lLowCut(iConfig.getUntrackedParameter<double>("m4lLowCut",0.0)),
  jetpt_cut(iConfig.getUntrackedParameter<double>("jetpt_cut",10.0)),
  jeteta_cut(iConfig.getUntrackedParameter<double>("eta_cut",4.7)),
  elecID(iConfig.getUntrackedParameter<std::string>("elecID","NonTrig")),
  isMC(iConfig.getUntrackedParameter<bool>("isMC",true)),
  isSignal(iConfig.getUntrackedParameter<bool>("isSignal",false)),
  mH(iConfig.getUntrackedParameter<double>("mH",0.0)),
  crossSection(iConfig.getUntrackedParameter<double>("CrossSection",1.0)),
  weightEvents(iConfig.getUntrackedParameter<bool>("weightEvents",false)),
  isoCutEl(iConfig.getUntrackedParameter<double>("isoCutEl",9999.0)),
  isoCutMu(iConfig.getUntrackedParameter<double>("isoCutMu",0.35)),/////ios is applied to new Muon BDT //previous 0.35///Qianying
  isoConeSizeEl(iConfig.getUntrackedParameter<double>("isoConeSizeEl",0.3)),
  isoConeSizeMu(iConfig.getUntrackedParameter<double>("isoConeSizeMu",0.3)),
  sip3dCut(iConfig.getUntrackedParameter<double>("sip3dCut",4)),
  leadingPtCut(iConfig.getUntrackedParameter<double>("leadingPtCut",20.0)),
  subleadingPtCut(iConfig.getUntrackedParameter<double>("subleadingPtCut",10.0)),
  genIsoCutEl(iConfig.getUntrackedParameter<double>("genIsoCutEl",0.35)), 
  genIsoCutMu(iConfig.getUntrackedParameter<double>("genIsoCutMu",0.35)), 
  genIsoConeSizeEl(iConfig.getUntrackedParameter<double>("genIsoConeSizeEl",0.3)), 
  genIsoConeSizeMu(iConfig.getUntrackedParameter<double>("genIsoConeSizeMu",0.3)), 
  _elecPtCut(iConfig.getUntrackedParameter<double>("_elecPtCut",7.0)),
  _muPtCut(iConfig.getUntrackedParameter<double>("_muPtCut",5.0)),
  _tauPtCut(iConfig.getUntrackedParameter<double>("_tauPtCut",20.0)),
  _phoPtCut(iConfig.getUntrackedParameter<double>("_phoPtCut",10.0)),
  //BTagCut(iConfig.getUntrackedParameter<double>("BTagCut",0.4184)),/////2016: 0.6321; 2017: 0.4941; 2018: 0.4184
  reweightForPU(iConfig.getUntrackedParameter<bool>("reweightForPU",true)),
  PUVersion(iConfig.getUntrackedParameter<std::string>("PUVersion","Summer16_80X")),
  doFsrRecovery(iConfig.getUntrackedParameter<bool>("doFsrRecovery",true)),
  bestCandMela(iConfig.getUntrackedParameter<bool>("bestCandMela",true)), //controlla
  doMela(iConfig.getUntrackedParameter<bool>("doMela",true)),  //controlla
  GENbestM4l(iConfig.getUntrackedParameter<bool>("GENbestM4l",false)),
  doPUJetID(iConfig.getUntrackedParameter<bool>("doPUJetID",true)),
  jetIDLevel(iConfig.getUntrackedParameter<int>("jetIDLevel",2)),
  doJER(iConfig.getUntrackedParameter<bool>("doJER",true)),
  doJEC(iConfig.getUntrackedParameter<bool>("doJEC",true)),
  doRefit(iConfig.getUntrackedParameter<bool>("doRefit",true)),
  doTriggerMatching(iConfig.getUntrackedParameter<bool>("doTriggerMatching",!isMC)),
  checkOnlySingle(iConfig.getUntrackedParameter<bool>("checkOnlySingle",false)),
  triggerList(iConfig.getUntrackedParameter<std::vector<std::string>>("triggerList")),
  skimLooseLeptons(iConfig.getUntrackedParameter<int>("skimLooseLeptons",2)),    
  skimTightLeptons(iConfig.getUntrackedParameter<int>("skimTightLeptons",2)),    
  verbose(iConfig.getUntrackedParameter<bool>("verbose",false)),
  year(iConfig.getUntrackedParameter<int>("year",2018)),////for year put 2016,2017, or 2018 to select correct training
  isCode4l(iConfig.getUntrackedParameter<bool>("isCode4l",true))    

  
  
  {
  
    if(!isMC){reweightForPU = false;}
    
    
  nEventsTotal=0.0;
  sumWeightsTotal=0.0;
  sumWeightsTotalPU=0.0;
  histContainer_["NEVENTS"]=fs->make<TH1F>("nEvents","nEvents in Sample",2,0,2);
  histContainer_["SUMWEIGHTS"]=fs->make<TH1F>("sumWeights","sum Weights of Sample",2,0,2);
  histContainer_["SUMWEIGHTSPU"]=fs->make<TH1F>("sumWeightsPU","sum Weights and PU of Sample",2,0,2);
  histContainer_["NVTX"]=fs->make<TH1F>("nVtx","Number of Vertices",36,-0.5,35.5);
  histContainer_["NVTX_RW"]=fs->make<TH1F>("nVtx_ReWeighted","Number of Vertices",36,-0.5,35.5);
  histContainer_["NINTERACT"]=fs->make<TH1F>("nInteractions","Number of True Interactions",61,-0.5,60.5);
  histContainer_["NINTERACT_RW"]=fs->make<TH1F>("nInteraction_ReWeighted","Number of True Interactions",61,-0.5,60.5);
  
  passedEventsTree_All = new TTree("passedEvents","passedEvents");
  
  edm::FileInPath kfacfileInPath("UFHZZAnalysisRun2/UFHZZ4LAna/data/Kfactor_ggHZZ_2l2l_NNLO_NNPDF_NarrowWidth_13TeV.root");
  TFile *fKFactor = TFile::Open(kfacfileInPath.fullPath().c_str());
  kFactor_ggzz = (TSpline3*) fKFactor->Get("sp_Kfactor");
  fKFactor->Close();
  delete fKFactor;

  tableEwk = readFile_and_loadEwkTable("ZZBG");   
  
  
  //string elec_scalefac_Cracks_name_161718[3] = {"egammaEffi.txt_EGM2D_cracks.root", "egammaEffi.txt_EGM2D_Moriond2018v1_gap.root", "egammaEffi.txt_EGM2D_Moriond2019_v1_gap.root"};
    string elec_scalefac_Cracks_name_161718[3] = {"ElectronSF_Legacy_2016_Gap.root", "ElectronSF_Legacy_2017_Gap.root", "ElectronSF_Legacy_2018_Gap.root"};
    edm::FileInPath elec_scalefacFileInPathCracks(("UFHZZAnalysisRun2/UFHZZ4LAna/data/"+elec_scalefac_Cracks_name_161718[year-2016]).c_str());
    TFile *fElecScalFacCracks = TFile::Open(elec_scalefacFileInPathCracks.fullPath().c_str());
    hElecScaleFac_Cracks = (TH2F*)fElecScalFacCracks->Get("EGamma_SF2D");    
    
    //string elec_scalefac_name_161718[3] = {"egammaEffi.txt_EGM2D.root", "egammaEffi.txt_EGM2D_Moriond2018v1.root", "egammaEffi.txt_EGM2D_Moriond2019_v1.root"};
    string elec_scalefac_name_161718[3] = {"ElectronSF_Legacy_2016_NoGap.root", "ElectronSF_Legacy_2017_NoGap.root", "ElectronSF_Legacy_2018_NoGap.root"};
    edm::FileInPath elec_scalefacFileInPath(("UFHZZAnalysisRun2/UFHZZ4LAna/data/"+elec_scalefac_name_161718[year-2016]).c_str());
    TFile *fElecScalFac = TFile::Open(elec_scalefacFileInPath.fullPath().c_str());
    hElecScaleFac = (TH2F*)fElecScalFac->Get("EGamma_SF2D");    

    //string elec_Gsfscalefac_name_161718[3] = {"egammaEffi.txt_EGM2D_GSF.root", "egammaEffi.txt_EGM2D_Moriond2018v1_runBCDEF_passingRECO.root", "Ele_Reco_2018.root"};//was previous;
    string elec_Gsfscalefac_name_161718[3] = {"Ele_Reco_2016.root", "Ele_Reco_2017.root", "Ele_Reco_2018.root"};
    edm::FileInPath elec_GsfscalefacFileInPath(("UFHZZAnalysisRun2/UFHZZ4LAna/data/"+elec_Gsfscalefac_name_161718[year-2016]).c_str());
    TFile *fElecScalFacGsf = TFile::Open(elec_GsfscalefacFileInPath.fullPath().c_str());
    hElecScaleFacGsf = (TH2F*)fElecScalFacGsf->Get("EGamma_SF2D");

    //string elec_GsfLowETscalefac_name_161718[3]= {"", "egammaEffi.txt_EGM2D_Moriond2018v1_runBCDEF_passingRECO_lowEt.root", "Ele_Reco_LowEt_2018.root"};//was previous
    string elec_GsfLowETscalefac_name_161718[3]= {"Ele_Reco_LowEt_2016.root", "Ele_Reco_LowEt_2017.root", "Ele_Reco_LowEt_2018.root"};
    edm::FileInPath elec_GsfLowETscalefacFileInPath(("UFHZZAnalysisRun2/UFHZZ4LAna/data/"+elec_GsfLowETscalefac_name_161718[year-2016]).c_str());
    TFile *fElecScalFacGsfLowET = TFile::Open(elec_GsfLowETscalefacFileInPath.fullPath().c_str());
    hElecScaleFacGsfLowET = (TH2F*)fElecScalFacGsfLowET->Get("EGamma_SF2D");

    //string mu_scalefac_name_161718[3] = {"final_HZZ_Moriond17Preliminary_v4.root", "ScaleFactors_mu_Moriond2018_final.root", "final_HZZ_muon_SF_2018RunA2D_ER_2702.root"};//was previous; 
//         string mu_scalefac_name_161718[3] = {"final_HZZ_SF_2016_legacy_mupogsysts.root", "final_HZZ_SF_2017_rereco_mupogsysts_3010.root", "final_HZZ_SF_2018_rereco_mupogsysts_3010.root"};
        string mu_scalefac_name_161718[3] = {"final_HZZ_muon_SF_2016RunB2H_legacy_newLoose_newIso_paper.root", "final_HZZ_muon_SF_2017_newLooseIso_mupogSysts_paper.root", "final_HZZ_muon_SF_2018RunA2D_ER_newLoose_newIso_paper.root"};
    edm::FileInPath mu_scalefacFileInPath(("UFHZZAnalysisRun2/UFHZZ4LAna/data/"+mu_scalefac_name_161718[year-2016]).c_str());
    TFile *fMuScalFac = TFile::Open(mu_scalefacFileInPath.fullPath().c_str());
    hMuScaleFac = (TH2F*)fMuScalFac->Get("FINAL");
    hMuScaleFacUnc = (TH2F*)fMuScalFac->Get("ERROR");

    //string pileup_name_161718[3] = {"puWeightsMoriond17_v2.root", "puWeightsMoriond18.root", "pu_weights_2018.root"};///was previous
    string pileup_name_161718[3] = {"pu_weights_2016.root", "pu_weights_2017.root", "pu_weights_2018.root"};
    edm::FileInPath pileup_FileInPath(("UFHZZAnalysisRun2/UFHZZ4LAna/data/"+pileup_name_161718[year-2016]).c_str());
    TFile *f_pileup = TFile::Open(pileup_FileInPath.fullPath().c_str());
    h_pileup = (TH1D*)f_pileup->Get("weights");
    h_pileupUp = (TH1D*)f_pileup->Get("weights_varUp");
    h_pileupDn = (TH1D*)f_pileup->Get("weights_varDn");

    string bTagEffi_name_161718[3] = {"bTagEfficiencies_2016.root", "bTagEfficiencies_2017.root", "bTagEfficiencies_2018.root"};
    edm::FileInPath BTagEffiInPath(("UFHZZAnalysisRun2/UFHZZ4LAna/data/"+bTagEffi_name_161718[year-2016]).c_str());
    TFile *fbTagEffi = TFile::Open(BTagEffiInPath.fullPath().c_str());
    hbTagEffi = (TH2F*)fbTagEffi->Get("eff_b_M_ALL");
    hcTagEffi = (TH2F*)fbTagEffi->Get("eff_c_M_ALL");
    hudsgTagEffi = (TH2F*)fbTagEffi->Get("eff_udsg_M_ALL");


    //BTag calibration
    string csv_name_161718[3] = {"DeepCSV_2016LegacySF_V1.csv", "DeepCSV_94XSF_V4_B_F.csv", "DeepCSV_102XSF_V1.csv"};
    edm::FileInPath btagfileInPath(("UFHZZAnalysisRun2/UFHZZ4LAna/data/"+csv_name_161718[year-2016]).c_str());

    BTagCalibration calib("DeepCSV", btagfileInPath.fullPath().c_str());
    reader = new BTagCalibrationReader(BTagEntry::OP_MEDIUM,  // operating point
                                       "central",             // central sys type
                                       {"up", "down"});      // other sys types
   

    reader->load(calib,                // calibration instance
                BTagEntry::FLAV_B,    // btag flavour
                "comb");               // measurement type

    if(year==2018)    {EleBDT_name_161718 = "ElectronMVAEstimatorRun2Autumn18IdIsoValues"; BTagCut=0.4184; heepID_name_161718 = "heepElectronID-HEEPV70";}
    if(year==2017)    {EleBDT_name_161718 = "ElectronMVAEstimatorRun2Fall17IsoV2Values"; BTagCut=0.4941; heepID_name_161718 = "heepElectronID-HEEPV70";}
    if(year==2016)    {EleBDT_name_161718 = "ElectronMVAEstimatorRun2Summer16IdIsoValues"; BTagCut=0.6321; heepID_name_161718 = "heepElectronID-HEEPV70";}


	std::string DATAPATH = std::getenv( "CMSSW_BASE" );
	if(year == 2018)    DATAPATH+="/src/UFHZZAnalysisRun2/KalmanMuonCalibrationsProducer/data/roccor.Run2.v3/RoccoR2018.txt";
	if(year == 2017)    DATAPATH+="/src/UFHZZAnalysisRun2/KalmanMuonCalibrationsProducer/data/roccor.Run2.v3/RoccoR2017.txt";
	if(year == 2016)    DATAPATH+="/src/UFHZZAnalysisRun2/KalmanMuonCalibrationsProducer/data/roccor.Run2.v3/RoccoR2016.txt";
	calibrator = new RoccoR(DATAPATH);

}


UFHZZ4LAna::~UFHZZ4LAna()
{
    //destructor --- don't do anything here  
}


/*float UFHZZ4LAna::ApplyRoccoR(int Y, bool isMC, int charge, float pt, float eta, float phi, float genPt, float nLayers){


	float scale_factor;
	if(isMC && nLayers > 5)
	{
		if(genPt > 0)
			scale_factor = calibrator->kSpreadMC(charge, pt, eta, phi, genPt);
		else{
			   TRandom3 rand;                                                                                                                                                                                                             
			   rand.SetSeed(abs(static_cast<int>(sin(phi)*100000)));                                                                                          

			   double u1;
			   u1 = rand.Uniform(1.);
			   scale_factor = calibrator->kSmearMC(charge, pt, eta, phi, nLayers, u1);
		}
	}
	else
		scale_factor = calibrator->kScaleDT(charge, pt, eta, phi);

	return scale_factor;

}*/

// ------------ method called for each event  ------------
void
UFHZZ4LAna::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

	using namespace edm;
  using namespace std;
  using namespace pat;
  using namespace trigger;
  using namespace EwkCorrections;

        
  nEventsTotal += 1.0;

  Run = iEvent.id().run();
  Event = iEvent.id().event();
  LumiSect = iEvent.id().luminosityBlock();

  if (verbose) {
     cout<<"Run: " << Run << ",Event: " << Event << ",LumiSect: "<<LumiSect<<endl;
  }

  // ======= Get Collections ======= //
  if (verbose) {cout<<"getting collections"<<endl;}

  // trigger collection
  edm::Handle<edm::TriggerResults> trigger;
  iEvent.getByToken(triggerSrc_,trigger);
  const edm::TriggerNames trigNames = iEvent.triggerNames(*trigger);

  // trigger Objects
  edm::Handle<pat::TriggerObjectStandAloneCollection> triggerObjects;
  iEvent.getByToken(triggerObjects_, triggerObjects);
  
  // met collection 
  edm::Handle<edm::View<pat::MET> > mets;
  iEvent.getByToken(metSrc_,mets);
  
  
  // Jets
  edm::Handle<edm::View<pat::Jet> > jets;
  iEvent.getByToken(jetSrc_,jets);

  if (!jecunc) {
      edm::ESHandle<JetCorrectorParametersCollection> jetCorrParameterSet;
      iSetup.get<JetCorrectionsRecord>().get("AK4PFchs", jetCorrParameterSet);
      const JetCorrectorParameters& jetCorrParameters = (*jetCorrParameterSet)["Uncertainty"];
      jecunc.reset(new JetCorrectionUncertainty(jetCorrParameters));
  }

  resolution_pt = JME::JetResolution::get(iSetup, "AK4PFchs_pt");
  resolution_phi = JME::JetResolution::get(iSetup, "AK4PFchs_phi");
  resolution_sf = JME::JetResolutionScaleFactor::get(iSetup, "AK4PFchs");

  edm::Handle<edm::ValueMap<float> > qgHandle;
  iEvent.getByToken(qgTagSrc_, qgHandle);

  edm::Handle<edm::ValueMap<float> > axis2Handle;
  iEvent.getByToken(axis2Src_, axis2Handle);

  edm::Handle<edm::ValueMap<int> > multHandle;
  iEvent.getByToken(multSrc_, multHandle);

  edm::Handle<edm::ValueMap<float> > ptDHandle;
  iEvent.getByToken(ptDSrc_, ptDHandle);
 
  edm::Handle<edm::View<pat::Jet> > mergedjets;
  iEvent.getByToken(mergedjetSrc_,mergedjets);
  
  edm::Handle<edm::View<reco::GenJet> > genJets;
  iEvent.getByToken(genJetsSrc_, genJets);
    
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(generatorSrc_,genEventInfo);
    
  //vector<edm::Handle<LHEEventProduct> > lheInfos;
  //iEvent.getManyByType(lheInfos); // using this method because the label is not always the same (e.g. "source" in the ttH sample)

  edm::Handle<LHEEventProduct> lheInfo;
  iEvent.getByToken(lheInfoSrc_, lheInfo);
  
  if (isMC) {    
        edm::Handle< double > theprefweight;
            iEvent.getByToken(prefweight_token_, theprefweight ) ;
            if (year == 2016 || year == 2017)
                prefiringWeight =(*theprefweight);
            else if (year == 2018)
                prefiringWeight =1.0;
    }
  else
  	prefiringWeight =1.0;
        
  if (verbose) {cout<<"clear variables"<<endl;}
  nVtx = -1.0; nInt = -1.0;
  finalState = -1;
  triggersPassed="";
  passedTrig=false;
  
  // Event Weights
  genWeight=1.0; pileupWeight=1.0; pileupWeightUp=1.0; pileupWeightDn=1.0; dataMCWeight=1.0; eventWeight=1.0;
  k_ggZZ=1.0; k_qqZZ_qcd_dPhi = 1.0; k_qqZZ_qcd_M = 1.0; k_qqZZ_qcd_Pt = 1.0; k_qqZZ_ewk = 1.0;

  qcdWeights.clear(); nnloWeights.clear(); pdfWeights.clear();
  pdfRMSup=1.0; pdfRMSdown=1.0; pdfENVup=1.0; pdfENVdown=1.0;
  
  
  // Jets
  jet_pt.clear(); jet_eta.clear(); jet_phi.clear(); jet_mass.clear(); jet_pt_raw.clear(); 
  jet_jesup_pt.clear(); jet_jesup_eta.clear(); jet_jesup_phi.clear(); jet_jesup_mass.clear(); 
  jet_jesdn_pt.clear(); jet_jesdn_eta.clear(); jet_jesdn_phi.clear(); jet_jesdn_mass.clear(); 
  jet_jerup_pt.clear(); jet_jerup_eta.clear(); jet_jerup_phi.clear(); jet_jerup_mass.clear(); 
  jet_jerdn_pt.clear(); jet_jerdn_eta.clear(); jet_jerdn_phi.clear(); jet_jerdn_mass.clear(); 
  jet_csvv2_.clear();
  jet_pumva.clear(); jet_csvv2.clear(); jet_isbtag.clear();
  jet_hadronFlavour.clear(); jet_partonFlavour.clear();
  jet_QGTagger.clear(); jet_QGTagger_jesup.clear(); jet_QGTagger_jesdn.clear(); 
  jet_relpterr.clear(); jet_phierr.clear();
  jet_bTagEffi.clear();
  jet_cTagEffi.clear();
  jet_udsgTagEffi.clear();
  jet_axis2.clear(); jet_ptD.clear(); jet_mult.clear();

  jet_iscleanH4l.clear();
  jet1index=-1; jet2index=-1;
  jet_jesup_iscleanH4l.clear(); jet_jesdn_iscleanH4l.clear(); 
  jet_jerup_iscleanH4l.clear(); jet_jerdn_iscleanH4l.clear();

  njets_pt30_eta4p7=0;
  njets_pt30_eta4p7_jesup=0; njets_pt30_eta4p7_jesdn=0;
  njets_pt30_eta4p7_jerup=0; njets_pt30_eta4p7_jerdn=0;

  njets_pt30_eta2p5=0;
  njets_pt30_eta2p5_jesup=0; njets_pt30_eta2p5_jesdn=0;
  njets_pt30_eta2p5_jerup=0; njets_pt30_eta2p5_jerdn=0;

  nbjets_pt30_eta4p7=0; nvjets_pt40_eta2p4=0;

  pt_leadingjet_pt30_eta4p7=-1.0;
  pt_leadingjet_pt30_eta4p7_jesup=-1.0; pt_leadingjet_pt30_eta4p7_jesdn=-1.0;
  pt_leadingjet_pt30_eta4p7_jerup=-1.0; pt_leadingjet_pt30_eta4p7_jerdn=-1.0;

  pt_leadingjet_pt30_eta2p5=-1.0;
  pt_leadingjet_pt30_eta2p5_jesup=-1.0; pt_leadingjet_pt30_eta2p5_jesdn=-1.0;
  pt_leadingjet_pt30_eta2p5_jerup=-1.0; pt_leadingjet_pt30_eta2p5_jerdn=-1.0;

  absrapidity_leadingjet_pt30_eta4p7=-1.0;
  absrapidity_leadingjet_pt30_eta4p7_jesup=-1.0; absrapidity_leadingjet_pt30_eta4p7_jesdn=-1.0;
  absrapidity_leadingjet_pt30_eta4p7_jerup=-1.0; absrapidity_leadingjet_pt30_eta4p7_jerdn=-1.0;

  absdeltarapidity_hleadingjet_pt30_eta4p7=-1.0;
  absdeltarapidity_hleadingjet_pt30_eta4p7_jesup=-1.0; absdeltarapidity_hleadingjet_pt30_eta4p7_jesdn=-1.0;
  absdeltarapidity_hleadingjet_pt30_eta4p7_jerup=-1.0; absdeltarapidity_hleadingjet_pt30_eta4p7_jerdn=-1.0;

  DijetMass=-1.0; DijetDEta=9999.0; DijetFisher=9999.0;
    
  mergedjet_iscleanH4l.clear();
  mergedjet_pt.clear(); mergedjet_eta.clear(); mergedjet_phi.clear(); mergedjet_mass.clear();
  mergedjet_L1.clear();
  mergedjet_softdropmass.clear(); mergedjet_prunedmass.clear();
  mergedjet_tau1.clear(); mergedjet_tau2.clear();
  mergedjet_btag.clear();

  mergedjet_nsubjet.clear();
  mergedjet_subjet_pt.clear(); mergedjet_subjet_eta.clear(); 
  mergedjet_subjet_phi.clear(); mergedjet_subjet_mass.clear();
  mergedjet_subjet_btag.clear();
  mergedjet_subjet_partonFlavour.clear(); mergedjet_subjet_hadronFlavour.clear();

	// -------------------------
  // GEN level information
  // ------------------------- 

  //Event variables
  GENfinalState=-1;
  passedFiducialSelection=false;
  
  // Jets
  GENjet_pt.clear(); GENjet_eta.clear(); GENjet_phi.clear(); GENjet_mass.clear(); 
  GENnjets_pt30_eta4p7=0;
  GENnjets_pt30_eta2p5=0;
  GENpt_leadingjet_pt30_eta4p7=-1.0; GENabsrapidity_leadingjet_pt30_eta4p7=-1.0; GENabsdeltarapidity_hleadingjet_pt30_eta4p7=-1.0;
  GENpt_leadingjet_pt30_eta2p5=-1.0; 
  lheNb=0; lheNj=0; nGenStatus2bHad=0;

   

  if (verbose) {cout<<"clear other variables"<<endl; }
  // Resolution
  //massErrorUCSD=-1.0; massErrorUCSDCorr=-1.0; massErrorUF=-1.0; massErrorUFCorr=-1.0; massErrorUFADCorr=-1.0;

  // Event Category
  EventCat=-1;
  
  jet1pt=-1.0; jet2pt=-1.0;
  
  jet_pt_float.clear(); jet_eta_float.clear(); jet_phi_float.clear(); jet_mass_float.clear(); jet_pt_raw_float.clear(); 
  jet_jesup_pt_float.clear(); jet_jesup_eta_float.clear(); jet_jesup_phi_float.clear(); jet_jesup_mass_float.clear();
  jet_jesdn_pt_float.clear(); jet_jesdn_eta_float.clear(); jet_jesdn_phi_float.clear(); jet_jesdn_mass_float.clear();
  jet_jerup_pt_float.clear(); jet_jerup_eta_float.clear(); jet_jerup_phi_float.clear(); jet_jerup_mass_float.clear();
  jet_jerdn_pt_float.clear(); jet_jerdn_eta_float.clear(); jet_jerdn_phi_float.clear();  jet_jerdn_mass_float.clear();
  
  
  // ====================== Do Analysis ======================== //
  
  
  if(isMC) {
  	if (verbose) cout<<"setting gen variables"<<endl;       
    setGENVariables(genJets); 
    if (verbose) { cout<<"finshed setting gen variables"<<endl;  }
  }
  
  eventWeight = 1;

  unsigned int _tSize = trigger->size();
  // create a string with all passing trigger names
  for (unsigned int i=0; i<_tSize; ++i) {
  	std::string triggerName = trigNames.triggerName(i);
    if (strstr(triggerName.c_str(),"_step")) continue;
    if (strstr(triggerName.c_str(),"MC_")) continue;
    if (strstr(triggerName.c_str(),"AlCa_")) continue;
    if (strstr(triggerName.c_str(),"DST_")) continue;
    if (strstr(triggerName.c_str(),"HLT_HI")) continue;
    if (strstr(triggerName.c_str(),"HLT_Physics")) continue;
    if (strstr(triggerName.c_str(),"HLT_Random")) continue;
    if (strstr(triggerName.c_str(),"HLT_ZeroBias")) continue;
    if (strstr(triggerName.c_str(),"HLT_IsoTrack")) continue;
    if (strstr(triggerName.c_str(),"Hcal")) continue;
    if (strstr(triggerName.c_str(),"Ecal")) continue;
    if (trigger->accept(i)) triggersPassed += triggerName; 
  }
  if (firstEntry) cout<<"triggersPassed: "<<triggersPassed<<endl;
  firstEntry = false;
  
  if (!mets->empty()) {
  	met = (*mets)[0].et();
    met_phi = (*mets)[0].phi();
    met_jesup = (*mets)[0].shiftedPt(pat::MET::JetEnUp);
   	met_phi_jesup = (*mets)[0].shiftedPhi(pat::MET::JetEnUp);
    met_jesdn = (*mets)[0].shiftedPt(pat::MET::JetEnDown);
    met_phi_jesdn = (*mets)[0].shiftedPhi(pat::MET::JetEnDown);
    met_uncenup = (*mets)[0].shiftedPt(pat::MET::UnclusteredEnUp);
    met_phi_uncenup = (*mets)[0].shiftedPhi(pat::MET::UnclusteredEnUp);
   	met_uncendn = (*mets)[0].shiftedPt(pat::MET::UnclusteredEnDown);
    met_phi_uncendn = (*mets)[0].shiftedPhi(pat::MET::UnclusteredEnDown);        
 	}
 	
 	
 	// Jets
 	if (verbose) cout<<"begin filling jet candidates"<<endl;
                
  vector<pat::Jet> goodJets;
  vector<float> patJetQGTagger, patJetaxis2, patJetptD;
  vector<float> goodJetQGTagger, goodJetaxis2, goodJetptD; 
  vector<int> patJetmult, goodJetmult;
                
  for(auto jet = jets->begin();  jet != jets->end(); ++jet){
  	edm::RefToBase<pat::Jet> jetRef(edm::Ref<edm::View<pat::Jet> >(jets, jet - jets->begin()));
   	float qgLikelihood = (*qgHandle)[jetRef];
    float axis2 = (*axis2Handle)[jetRef];
    float ptD = (*ptDHandle)[jetRef];
    int mult = (*multHandle)[jetRef];
    patJetQGTagger.push_back(qgLikelihood);  
    patJetaxis2.push_back(axis2);  
    patJetmult.push_back(mult);  
    patJetptD.push_back(ptD);  
  }
                           
	for(unsigned int i = 0; i < jets->size(); ++i) {
                    
	 	const pat::Jet & jet = jets->at(i);
		                  
		//JetID ID
		if (verbose) cout<<"checking jetid..."<<endl;
		float jpumva=0.;
		bool passPU;
	 	if (doJEC && (year==2017 || year==2018)) {
			passPU = bool(jet.userInt("pileupJetIdUpdated:fullId") & (1 << 0));
		  jpumva=jet.userFloat("pileupJetIdUpdated:fullDiscriminant");
		} 
		else {
			passPU = bool(jet.userInt("pileupJetId:fullId") & (1 << 0));
		  jpumva=jet.userFloat("pileupJetId:fullDiscriminant");
		}
		if (verbose) cout<< " jet pu mva  "<<jpumva <<endl;
	 
		            
		        
		if (verbose) cout<<"pt: "<<jet.pt()<<" eta: "<<jet.eta()<<" phi: "<<jet.phi()<<" passPU: "<<passPU
		                 <<" jetid: "<<jetHelper.patjetID(jet,year)<<endl;
		    
		if( jetHelper.patjetID(jet,year)>=jetIDLevel ) {       
			if (verbose) cout<<"passed pf jet id and pu jet id"<<endl;
			if (verbose) cout<<"adding jet candidate, pt: "<<jet.pt()<<" eta: "<<jet.eta()<<endl;
		  goodJets.push_back(jet);
		  goodJetQGTagger.push_back(patJetQGTagger[i]);
		  goodJetaxis2.push_back(patJetaxis2[i]);
		  goodJetptD.push_back(patJetptD[i]);
		  goodJetmult.push_back(patJetmult[i]);
		}
  }
  
  vector<pat::Jet> selectedMergedJets;
  
  
  if (verbose) cout<<"begin setting tree variables"<<endl;
	setTreeVariables(iEvent, iSetup, goodJets, goodJetQGTagger,goodJetaxis2, goodJetptD, goodJetmult, selectedMergedJets);
  if (verbose) cout<<"finshed setting tree variables"<<endl;
  
  jet_pt_float.assign(jet_pt.begin(),jet_pt.end());
  jet_pt_raw_float.assign(jet_pt_raw.begin(),jet_pt_raw.end());
  jet_eta_float.assign(jet_eta.begin(),jet_eta.end());
  jet_phi_float.assign(jet_phi.begin(),jet_phi.end());
  jet_mass_float.assign(jet_mass.begin(),jet_mass.end());
  jet_jesup_pt_float.assign(jet_jesup_pt.begin(),jet_jesup_pt.end());
  jet_jesup_eta_float.assign(jet_jesup_eta.begin(),jet_jesup_eta.end());
  jet_jesup_phi_float.assign(jet_jesup_phi.begin(),jet_jesup_phi.end());
  jet_jesup_mass_float.assign(jet_jesup_mass.begin(),jet_jesup_mass.end());
  jet_jesdn_pt_float.assign(jet_jesdn_pt.begin(),jet_jesdn_pt.end());
  jet_jesdn_eta_float.assign(jet_jesdn_eta.begin(),jet_jesdn_eta.end());
  jet_jesdn_phi_float.assign(jet_jesdn_phi.begin(),jet_jesdn_phi.end());
  jet_jesdn_mass_float.assign(jet_jesdn_mass.begin(),jet_jesdn_mass.end());
  jet_jerup_pt_float.assign(jet_jerup_pt.begin(),jet_jerup_pt.end());
  jet_jerup_eta_float.assign(jet_jerup_eta.begin(),jet_jerup_eta.end());
  jet_jerup_phi_float.assign(jet_jerup_phi.begin(),jet_jerup_phi.end());
  jet_jerup_mass_float.assign(jet_jerup_mass.begin(),jet_jerup_mass.end());
  jet_jerdn_pt_float.assign(jet_jerdn_pt.begin(),jet_jerdn_pt.end());
  jet_jerdn_eta_float.assign(jet_jerdn_eta.begin(),jet_jerdn_eta.end());
  jet_jerdn_phi_float.assign(jet_jerdn_phi.begin(),jet_jerdn_phi.end());
  jet_jerdn_mass_float.assign(jet_jerdn_mass.begin(),jet_jerdn_mass.end());
  
  if (!isMC) passedEventsTree_All->Fill(); 
  
  GENjet_pt_float.clear(); GENjet_pt_float.assign(GENjet_pt.begin(),GENjet_pt.end());
  GENjet_eta_float.clear(); GENjet_eta_float.assign(GENjet_eta.begin(),GENjet_eta.end());
  GENjet_phi_float.clear(); GENjet_phi_float.assign(GENjet_phi.begin(),GENjet_phi.end());
  GENjet_mass_float.clear(); GENjet_mass_float.assign(GENjet_mass.begin(),GENjet_mass.end());

  if (isMC) passedEventsTree_All->Fill();
  
  if (nEventsTotal==1000.0) passedEventsTree_All->OptimizeBaskets();
  
  } //end UFHZZ4LAna::analyze
  

// ------------ method called once each job just before starting event loop  ------------
void 
UFHZZ4LAna::beginJob()
{
    using namespace edm;
    using namespace std;
    using namespace pat;

    bookPassedEventTree("passedEvents", passedEventsTree_All);

    firstEntry = true;

}

// ------------ method called once each job just after ending the event loop  ------------
void 
UFHZZ4LAna::endJob() 
{
    histContainer_["NEVENTS"]->SetBinContent(1,nEventsTotal);
    histContainer_["NEVENTS"]->GetXaxis()->SetBinLabel(1,"N Events in Sample");
    histContainer_["SUMWEIGHTS"]->SetBinContent(1,sumWeightsTotal);
    histContainer_["SUMWEIGHTSPU"]->SetBinContent(1,sumWeightsTotalPU);
    histContainer_["SUMWEIGHTS"]->GetXaxis()->SetBinLabel(1,"sum Weights in Sample");
    histContainer_["SUMWEIGHTSPU"]->GetXaxis()->SetBinLabel(1,"sum Weights PU in Sample");
}

void
UFHZZ4LAna::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

    //massErr.init(iSetup);
    if (isMC) {
        edm::Handle<LHERunInfoProduct> run;
        typedef std::vector<LHERunInfoProduct::Header>::const_iterator headers_const_iterator;
        try {

            int pos=0;
            iRun.getByLabel( edm::InputTag("externalLHEProducer"), run );
            LHERunInfoProduct myLHERunInfoProduct = *(run.product());
            typedef std::vector<LHERunInfoProduct::Header>::const_iterator headers_const_iterator;
            for (headers_const_iterator iter=myLHERunInfoProduct.headers_begin(); iter!=myLHERunInfoProduct.headers_end(); iter++){
                std::cout << iter->tag() << std::endl;
                std::vector<std::string> lines = iter->lines();
                for (unsigned int iLine = 0; iLine<lines.size(); iLine++) {
                    std::string pdfid=lines.at(iLine);
                    if (pdfid.substr(1,6)=="weight" && pdfid.substr(8,2)=="id") {
                        std::cout<<pdfid<<std::endl;
                        std::string pdf_weight_id = pdfid.substr(12,4);
                        int pdf_weightid=atoi(pdf_weight_id.c_str());
                        std::cout<<"parsed id: "<<pdf_weightid<<std::endl;
                        if (pdf_weightid==2001) {posNNPDF=int(pos);}
                        pos+=1;
                    }
                }
            }
        }
        catch(...) {
            std::cout<<"No LHERunInfoProduct"<<std::endl;
        }
    }

}


// ------------ method called when ending the processing of a run  ------------
void 
UFHZZ4LAna::endRun(const edm::Run& iRun, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
UFHZZ4LAna::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
UFHZZ4LAna::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,edm::EventSetup const& eSetup)
{
    using namespace edm;
    using namespace std;
    // Keep track of all the events run over
    edm::Handle<MergeableCounter> numEventsCounter;
    lumiSeg.getByLabel("nEventsTotal", numEventsCounter);    
    if(numEventsCounter.isValid()) {
        std::cout<<"numEventsCounter->value "<<numEventsCounter->value<<endl;
        nEventsTotal += numEventsCounter->value;        
    }
}

// ============================ UF Functions ============================= //


//Find Z1,Z2, and Higgs candidate
//Pass good leptons for analysis as candMuons and candElectrons
//Pass empty vectors of pat leptons as selectedMuons and selectedElectrons
// these will be filled in the function and then useable for more analysis.



void UFHZZ4LAna::bookPassedEventTree(TString treeName, TTree *tree)
{     


    using namespace edm;
    using namespace pat;
    using namespace std;

    // -------------------------                                                                                                                                                                        
    // RECO level information                                                                                                                                                                           
    // -------------------------                                                                                                                                                                        
    // Event variables
    tree->Branch("Run",&Run,"Run/l");
    tree->Branch("Event",&Event,"Event/l");
    tree->Branch("LumiSect",&LumiSect,"LumiSect/l");
    /*tree->Branch("nVtx",&nVtx,"nVtx/I");
    tree->Branch("nInt",&nInt,"nInt/I");
    tree->Branch("PV_x", &PV_x, "PV_x/F");
    tree->Branch("PV_y", &PV_y, "PV_y/F");
    tree->Branch("PV_z", &PV_z, "PV_z/F");
    tree->Branch("BS_x", &BS_x, "BS_x/F");
    tree->Branch("BS_y", &BS_y, "BS_y/F");
    tree->Branch("BS_z", &BS_z, "BS_z/F");
    tree->Branch("BS_xErr", &BS_xErr, "BS_xErr/F");
    tree->Branch("BS_yErr", &BS_yErr, "BS_yErr/F");
    tree->Branch("BS_zErr", &BS_zErr, "BS_zErr/F");
    tree->Branch("BeamWidth_x", &BeamWidth_x, "BeamWidth_x/F");
    tree->Branch("BeamWidth_y", &BeamWidth_y, "BeamWidth_y/F");
    tree->Branch("BeamWidth_xErr", &BeamWidth_xErr, "BeamWidth_xErr/F");
    tree->Branch("BeamWidth_yErr", &BeamWidth_yErr, "BeamWidth_yErr/F");
    tree->Branch("finalState",&finalState,"finalState/I");
    tree->Branch("triggersPassed",&triggersPassed);
    tree->Branch("passedTrig",&passedTrig,"passedTrig/O");
    tree->Branch("passedFullSelection",&passedFullSelection,"passedFullSelection/O");
    tree->Branch("passedZ4lSelection",&passedZ4lSelection,"passedZ4lSelection/O");
    tree->Branch("passedQCDcut",&passedQCDcut,"passedQCDcut/O");
    tree->Branch("passedZ1LSelection",&passedZ1LSelection,"passedZ1LSelection/O");
    tree->Branch("passedZ4lZ1LSelection",&passedZ4lZ1LSelection,"passedZ4lZ1LSelection/O");
    tree->Branch("passedZ4lZXCRSelection",&passedZ4lZXCRSelection,"passedZ4lZXCRSelection/O");
    tree->Branch("passedZXCRSelection",&passedZXCRSelection,"passedZXCRSelection/O");
    tree->Branch("nZXCRFailedLeptons",&nZXCRFailedLeptons,"nZXCRFailedLeptons/I");
    tree->Branch("genWeight",&genWeight,"genWeight/F");
    tree->Branch("k_ggZZ",&k_ggZZ,"k_ggZZ/F");
    tree->Branch("k_qqZZ_qcd_dPhi",&k_qqZZ_qcd_dPhi,"k_qqZZ_qcd_dPhi/F");
    tree->Branch("k_qqZZ_qcd_M",&k_qqZZ_qcd_M,"k_qqZZ_qcd_M/F");
    tree->Branch("k_qqZZ_qcd_Pt",&k_qqZZ_qcd_Pt,"k_qqZZ_qcd_Pt/F");
    tree->Branch("k_qqZZ_ewk",&k_qqZZ_ewk,"k_qqZZ_ewk/F");
    tree->Branch("qcdWeights",&qcdWeights);
    tree->Branch("nnloWeights",&nnloWeights);
    tree->Branch("pdfWeights",&pdfWeights);
    tree->Branch("pdfRMSup",&pdfRMSup,"pdfRMSup/F");
    tree->Branch("pdfRMSdown",&pdfRMSdown,"pdfRMSdown/F");
    tree->Branch("pdfENVup",&pdfENVup,"pdfENVup/F");
    tree->Branch("pdfENVdown",&pdfENVdown,"pdfENVdown/F");
    tree->Branch("pileupWeight",&pileupWeight,"pileupWeight/F");
    tree->Branch("pileupWeightUp",&pileupWeightUp,"pileupWeightUp/F");
    tree->Branch("pileupWeightDn",&pileupWeightDn,"pileupWeightDn/F");
    tree->Branch("dataMCWeight",&dataMCWeight,"dataMCWeight/F");
    tree->Branch("eventWeight",&eventWeight,"eventWeight/F");
    tree->Branch("prefiringWeight",&prefiringWeight,"prefiringWeight/F");
    tree->Branch("crossSection",&crossSection,"crossSection/F");

    // Lepton variables
    tree->Branch("lep_d0BS",&lep_d0BS_float);
    tree->Branch("lep_d0PV",&lep_d0PV_float);

    tree->Branch("lep_numberOfValidPixelHits",&lep_numberOfValidPixelHits_float);
    tree->Branch("lep_trackerLayersWithMeasurement",&lep_trackerLayersWithMeasurement_float);

    tree->Branch("lep_p",&lep_p_float);
    tree->Branch("lep_ecalEnergy",&lep_ecalEnergy_float);
    tree->Branch("lep_isEB",&lep_isEB);
    tree->Branch("lep_isEE",&lep_isEE);


    tree->Branch("lep_pt_UnS",&lep_pt_UnS_float);
    tree->Branch("lep_pterrold_UnS",&lep_pterrold_UnS_float);
    tree->Branch("lep_errPre_Scale",&lep_errPre_Scale_float);
    tree->Branch("lep_errPost_Scale",&lep_errPost_Scale_float);
    tree->Branch("lep_errPre_noScale",&lep_errPre_noScale_float);
    tree->Branch("lep_errPost_noScale",&lep_errPost_noScale_float);


    tree->Branch("lep_pt_FromMuonBestTrack",&lep_pt_FromMuonBestTrack_float);
    tree->Branch("lep_eta_FromMuonBestTrack",&lep_eta_FromMuonBestTrack_float);
    tree->Branch("lep_phi_FromMuonBestTrack",&lep_phi_FromMuonBestTrack_float);
    
    tree->Branch("lep_position_x",&lep_position_x_float);
    tree->Branch("lep_position_y",&lep_position_y_float);
    tree->Branch("lep_position_z",&lep_position_z_float);
    tree->Branch("lep_pt_genFromReco",&lep_pt_genFromReco_float);

    tree->Branch("lep_id",&lep_id);
    tree->Branch("lep_pt",&lep_pt_float);
    tree->Branch("lep_pterr",&lep_pterr_float);
    tree->Branch("lep_pterrold",&lep_pterrold_float);
    tree->Branch("lep_eta",&lep_eta_float);
    tree->Branch("lep_phi",&lep_phi_float);
    tree->Branch("lep_mass",&lep_mass_float);
    tree->Branch("lepFSR_pt",&lepFSR_pt_float);
    tree->Branch("lepFSR_eta",&lepFSR_eta_float);
    tree->Branch("lepFSR_phi",&lepFSR_phi_float);
    tree->Branch("lepFSR_mass",&lepFSR_mass_float);
    tree->Branch("lep_Hindex",&lep_Hindex,"lep_Hindex[4]/I");
    tree->Branch("lep_genindex",&lep_genindex);
    tree->Branch("lep_matchedR03_PdgId",&lep_matchedR03_PdgId);
    tree->Branch("lep_matchedR03_MomId",&lep_matchedR03_MomId);
    tree->Branch("lep_matchedR03_MomMomId",&lep_matchedR03_MomMomId);
    tree->Branch("lep_missingHits",&lep_missingHits);
    tree->Branch("lep_mva",&lep_mva);
    tree->Branch("lep_ecalDriven",&lep_ecalDriven);
    tree->Branch("lep_tightId",&lep_tightId);
    //tree->Branch("lep_tightId_old",&lep_tightId_old);
    tree->Branch("lep_tightIdSUS",&lep_tightIdSUS);
    tree->Branch("lep_tightIdHiPt",&lep_tightIdHiPt);
    tree->Branch("lep_Sip",&lep_Sip);
    tree->Branch("lep_IP",&lep_IP);
    tree->Branch("lep_isoNH",&lep_isoNH);
    tree->Branch("lep_isoCH",&lep_isoCH);
    tree->Branch("lep_isoPhot",&lep_isoPhot);
    tree->Branch("lep_isoPU",&lep_isoPU);
    tree->Branch("lep_isoPUcorr",&lep_isoPUcorr);
    tree->Branch("lep_RelIso",&lep_RelIso);
    tree->Branch("lep_RelIsoNoFSR",&lep_RelIsoNoFSR);
    tree->Branch("lep_MiniIso",&lep_MiniIso);
    tree->Branch("lep_ptRatio",&lep_ptRatio);
    tree->Branch("lep_ptRel",&lep_ptRel);
    tree->Branch("lep_filtersMatched",&lep_filtersMatched);
    tree->Branch("lep_dataMC",&lep_dataMC);
    tree->Branch("lep_dataMCErr",&lep_dataMCErr);
    tree->Branch("dataMC_VxBS",&dataMC_VxBS);
    tree->Branch("dataMCErr_VxBS",&dataMCErr_VxBS);
    tree->Branch("nisoleptons",&nisoleptons,"nisoleptons/I");
    tree->Branch("muRho",&muRho,"muRho/F");
    tree->Branch("elRho",&elRho,"elRho/F");
    tree->Branch("pTL1",&pTL1,"pTL1/F");
    tree->Branch("pTL2",&pTL2,"pTL2/F");
    tree->Branch("pTL3",&pTL3,"pTL3/F");
    tree->Branch("pTL4",&pTL4,"pTL4/F");
    tree->Branch("idL1",&idL1,"idL1/I");
    tree->Branch("idL2",&idL2,"idL2/I");
    tree->Branch("idL3",&idL3,"idL3/I");
    tree->Branch("idL4",&idL4,"idL4/I");
    tree->Branch("etaL1",&etaL1,"etaL1/F");
    tree->Branch("etaL2",&etaL2,"etaL2/F");
    tree->Branch("etaL3",&etaL3,"etaL3/F");
    tree->Branch("etaL4",&etaL4,"etaL4/F");
    tree->Branch("mL1",&mL1,"mL1/F");
    tree->Branch("mL2",&mL2,"mL2/F");
    tree->Branch("mL3",&mL3,"mL3/F");
    tree->Branch("mL4",&mL4,"mL4/F");
    tree->Branch("pTErrL1",&pTErrL1,"pTErrL1/F");
    tree->Branch("pTErrL2",&pTErrL2,"pTErrL2/F");
    tree->Branch("pTErrL3",&pTErrL3,"pTErrL3/F");
    tree->Branch("pTErrL4",&pTErrL4,"pTErrL4/F");
    tree->Branch("phiL1",&phiL1,"phiL1/F");
    tree->Branch("phiL2",&phiL2,"phiL2/F");
    tree->Branch("phiL3",&phiL3,"phiL3/F");
    tree->Branch("phiL4",&phiL4,"phiL4/F");
    tree->Branch("pTL1FSR",&pTL1FSR,"pTL1FSR/F");
    tree->Branch("pTL2FSR",&pTL2FSR,"pTL2FSR/F");
    tree->Branch("pTL3FSR",&pTL3FSR,"pTL3FSR/F");
    tree->Branch("pTL4FSR",&pTL4FSR,"pTL4FSR/F");
    tree->Branch("etaL1FSR",&etaL1FSR,"etaL1FSR/F");
    tree->Branch("etaL2FSR",&etaL2FSR,"etaL2FSR/F");
    tree->Branch("etaL3FSR",&etaL3FSR,"etaL3FSR/F");
    tree->Branch("etaL4FSR",&etaL4FSR,"etaL4FSR/F");
    tree->Branch("phiL1FSR",&phiL1FSR,"phiL1FSR/F");
    tree->Branch("phiL2FSR",&phiL2FSR,"phiL2FSR/F");
    tree->Branch("phiL3FSR",&phiL3FSR,"phiL3FSR/F");
    tree->Branch("phiL4FSR",&phiL4FSR,"phiL4FSR/F");
    tree->Branch("mL1FSR",&mL1FSR,"mL1FSR/F");
    tree->Branch("mL2FSR",&mL2FSR,"mL2FSR/F");
    tree->Branch("mL3FSR",&mL3FSR,"mL3FSR/F");
    tree->Branch("mL4FSR",&mL4FSR,"mL4FSR/F");
    tree->Branch("pTErrL1FSR",&pTErrL1FSR,"pTErrL1FSR/F");
    tree->Branch("pTErrL2FSR",&pTErrL2FSR,"pTErrL2FSR/F");
    tree->Branch("pTErrL3FSR",&pTErrL3FSR,"pTErrL3FSR/F");
    tree->Branch("pTErrL4FSR",&pTErrL4FSR,"pTErrL4FSR/F");
    tree->Branch("tau_id",&tau_id);
    tree->Branch("tau_pt",&tau_pt_float);
    tree->Branch("tau_eta",&tau_eta_float);
    tree->Branch("tau_phi",&tau_phi_float);
    tree->Branch("tau_mass",&tau_mass_float);
    tree->Branch("pho_pt",&pho_pt_float);
    tree->Branch("pho_eta",&pho_eta_float);
    tree->Branch("pho_phi",&pho_phi_float);
    tree->Branch("photonCutBasedIDLoose",&photonCutBasedIDLoose_float);

    //Higgs Candidate Variables
    tree->Branch("H_pt",&H_pt_float);
    tree->Branch("H_eta",&H_eta_float);
    tree->Branch("H_phi",&H_phi_float);
    tree->Branch("H_mass",&H_mass_float);
    tree->Branch("H_noFSR_pt",&H_noFSR_pt_float);
    tree->Branch("H_noFSR_eta",&H_noFSR_eta_float);
    tree->Branch("H_noFSR_phi",&H_noFSR_phi_float);
    tree->Branch("H_noFSR_mass",&H_noFSR_mass_float);
    tree->Branch("mass4l",&mass4l,"mass4l/F");
    tree->Branch("mass4l_noFSR",&mass4l_noFSR,"mass4l_noFSR/F");
    
    
    
    

    
    

    tree->Branch("mass4mu",&mass4mu,"mass4mu/F");
    tree->Branch("mass4e",&mass4e,"mass4e/F");
    tree->Branch("mass2e2mu",&mass2e2mu,"mass2e2mu/F");
    tree->Branch("pT4l",&pT4l,"pT4l/F");
    tree->Branch("eta4l",&eta4l,"eta4l/F");
    tree->Branch("phi4l",&phi4l,"phi4l/F");
    tree->Branch("rapidity4l",&rapidity4l,"rapidity4l/F");

    // Z candidate variables
    tree->Branch("massZ_vtx_chi2_BS",&massZ_vtx_chi2_BS,"massZ_vtx_chi2_BS/F");
    tree->Branch("massZ_vtx_chi2",&massZ_vtx_chi2,"massZ_vtx_chi2/F");
    tree->Branch("mass2l_vtx",&mass2l_vtx,"mass2l_vtx/F");
    tree->Branch("mass2l_vtx_BS",&mass2l_vtx_BS,"mass2l_vtx_BS/F");

    tree->Branch("Z_pt",&Z_pt_float);
    tree->Branch("Z_eta",&Z_eta_float);
    tree->Branch("Z_phi",&Z_phi_float);
    tree->Branch("Z_mass",&Z_mass_float);
    tree->Branch("Z_noFSR_pt",&Z_noFSR_pt_float);
    tree->Branch("Z_noFSR_eta",&Z_noFSR_eta_float);
    tree->Branch("Z_noFSR_phi",&Z_noFSR_phi_float);
    tree->Branch("Z_noFSR_mass",&Z_noFSR_mass_float);
    tree->Branch("Z_Hindex",&Z_Hindex,"Z_Hindex[2]/I");
    tree->Branch("massZ1",&massZ1,"massZ1/F");
    tree->Branch("massErrH_vtx",&massErrH_vtx,"massErrH_vtx/F");
    tree->Branch("massH_vtx_chi2_BS",&massH_vtx_chi2_BS,"massH_vtx_chi2_BS/F");
    tree->Branch("massH_vtx_chi2",&massH_vtx_chi2,"massH_vtx_chi2/F");
    tree->Branch("massZ1_Z1L",&massZ1_Z1L,"massZ1_Z1L/F");
    tree->Branch("massZ2",&massZ2,"massZ2/F");  
    tree->Branch("pTZ1",&pTZ1,"pTZ1/F");
    tree->Branch("pTZ2",&pTZ2,"pTZ2/F");*/

    // MET
    tree->Branch("met",&met,"met/F");
    tree->Branch("met_phi",&met_phi,"met_phi/F");
    tree->Branch("met_jesup",&met_jesup,"met_jesup/F");
    tree->Branch("met_phi_jesup",&met_phi_jesup,"met_phi_jesup/F");
    tree->Branch("met_jesdn",&met_jesdn,"met_jesdn/F");
    tree->Branch("met_phi_jesdn",&met_phi_jesdn,"met_phi_jesdn/F");
    tree->Branch("met_uncenup",&met_uncenup,"met_uncenup/F");
    tree->Branch("met_phi_uncenup",&met_phi_uncenup,"met_phi_uncenup/F");
    tree->Branch("met_uncendn",&met_uncendn,"met_uncendn/F");
    tree->Branch("met_phi_uncendn",&met_phi_uncendn,"met_phi_uncendn/F");

    // Jets
    tree->Branch("jet_iscleanH4l",&jet_iscleanH4l);
    tree->Branch("jet1index",&jet1index,"jet1index/I");
    tree->Branch("jet2index",&jet2index,"jet2index/I");
    tree->Branch("jet_pt",&jet_pt_float);
    tree->Branch("jet_pt_raw",&jet_pt_raw_float);
    tree->Branch("jet_relpterr",&jet_relpterr);    
    tree->Branch("jet_eta",&jet_eta_float);
    tree->Branch("jet_phi",&jet_phi_float);
    tree->Branch("jet_phierr",&jet_phierr);
    /*tree->Branch("jet_bTagEffi",&jet_bTagEffi);
    tree->Branch("jet_cTagEffi",&jet_cTagEffi);
    tree->Branch("jet_udsgTagEffi",&jet_udsgTagEffi);
    tree->Branch("jet_mass",&jet_mass_float);    
    tree->Branch("jet_jesup_iscleanH4l",&jet_jesup_iscleanH4l);
    tree->Branch("jet_jesup_pt",&jet_jesup_pt_float);
    tree->Branch("jet_jesup_eta",&jet_jesup_eta_float);
    tree->Branch("jet_jesup_phi",&jet_jesup_phi_float);
    tree->Branch("jet_jesup_mass",&jet_jesup_mass_float);
    tree->Branch("jet_jesdn_iscleanH4l",&jet_jesdn_iscleanH4l);
    tree->Branch("jet_jesdn_pt",&jet_jesdn_pt_float);
    tree->Branch("jet_jesdn_eta",&jet_jesdn_eta_float);
    tree->Branch("jet_jesdn_phi",&jet_jesdn_phi_float);
    tree->Branch("jet_jesdn_mass",&jet_jesdn_mass_float);
    tree->Branch("jet_jerup_iscleanH4l",&jet_jerup_iscleanH4l);
    tree->Branch("jet_jerup_pt",&jet_jerup_pt_float);
    tree->Branch("jet_jerup_eta",&jet_jerup_eta_float);
    tree->Branch("jet_jerup_phi",&jet_jerup_phi_float);
    tree->Branch("jet_jerup_mass",&jet_jerup_mass_float);
    tree->Branch("jet_jerdn_iscleanH4l",&jet_jerdn_iscleanH4l);
    tree->Branch("jet_jerdn_pt",&jet_jerdn_pt_float);
    tree->Branch("jet_jerdn_eta",&jet_jerdn_eta_float);
    tree->Branch("jet_jerdn_phi",&jet_jerdn_phi_float);
    tree->Branch("jet_jerdn_mass",&jet_jerdn_mass_float);
    tree->Branch("jet_pumva",&jet_pumva);
    tree->Branch("jet_csvv2",&jet_csvv2);
    tree->Branch("jet_csvv2_",&jet_csvv2_);
    tree->Branch("jet_isbtag",&jet_isbtag);
    tree->Branch("jet_hadronFlavour",&jet_hadronFlavour);
    tree->Branch("jet_partonFlavour",&jet_partonFlavour);    
    tree->Branch("jet_QGTagger",&jet_QGTagger);
    tree->Branch("jet_QGTagger_jesup",&jet_QGTagger_jesup);
    tree->Branch("jet_QGTagger_jesdn",&jet_QGTagger_jesdn);
    tree->Branch("jet_axis2",&jet_axis2);
    tree->Branch("jet_ptD",&jet_ptD);
    tree->Branch("jet_mult",&jet_mult);
    tree->Branch("njets_pt30_eta4p7",&njets_pt30_eta4p7,"njets_pt30_eta4p7/I");
    tree->Branch("njets_pt30_eta4p7_jesup",&njets_pt30_eta4p7_jesup,"njets_pt30_eta4p7_jesup/I");
    tree->Branch("njets_pt30_eta4p7_jesdn",&njets_pt30_eta4p7_jesdn,"njets_pt30_eta4p7_jesdn/I");
    tree->Branch("njets_pt30_eta4p7_jerup",&njets_pt30_eta4p7_jerup,"njets_pt30_eta4p7_jerup/I");
    tree->Branch("njets_pt30_eta4p7_jerdn",&njets_pt30_eta4p7_jerdn,"njets_pt30_eta4p7_jerdn/I");
    tree->Branch("pt_leadingjet_pt30_eta4p7",&pt_leadingjet_pt30_eta4p7,"pt_leadingjet_pt30_eta4p7/F");
    tree->Branch("pt_leadingjet_pt30_eta4p7_jesup",&pt_leadingjet_pt30_eta4p7_jesup,"pt_leadingjet_pt30_eta4p7_jesup/F");
    tree->Branch("pt_leadingjet_pt30_eta4p7_jesdn",&pt_leadingjet_pt30_eta4p7_jesdn,"pt_leadingjet_pt30_eta4p7_jesdn/F");
    tree->Branch("pt_leadingjet_pt30_eta4p7_jerup",&pt_leadingjet_pt30_eta4p7_jerup,"pt_leadingjet_pt30_eta4p7_jerup/F");
    tree->Branch("pt_leadingjet_pt30_eta4p7_jerdn",&pt_leadingjet_pt30_eta4p7_jerdn,"pt_leadingjet_pt30_eta4p7_jerdn/F");
    tree->Branch("absrapidity_leadingjet_pt30_eta4p7",&absrapidity_leadingjet_pt30_eta4p7,"absrapidity_leadingjet_pt30_eta4p7/F");
    tree->Branch("absrapidity_leadingjet_pt30_eta4p7_jesup",&absrapidity_leadingjet_pt30_eta4p7_jesup,"absrapidity_leadingjet_pt30_eta4p7_jesup/F");
    tree->Branch("absrapidity_leadingjet_pt30_eta4p7_jesdn",&absrapidity_leadingjet_pt30_eta4p7_jesdn,"absrapidity_leadingjet_pt30_eta4p7_jesdn/F");
    tree->Branch("absrapidity_leadingjet_pt30_eta4p7_jerup",&absrapidity_leadingjet_pt30_eta4p7_jerup,"absrapidity_leadingjet_pt30_eta4p7_jerup/F");
    tree->Branch("absrapidity_leadingjet_pt30_eta4p7_jerdn",&absrapidity_leadingjet_pt30_eta4p7_jerdn,"absrapidity_leadingjet_pt30_eta4p7_jerdn/F");
    tree->Branch("absdeltarapidity_hleadingjet_pt30_eta4p7",&absdeltarapidity_hleadingjet_pt30_eta4p7,"absdeltarapidity_hleadingjet_pt30_eta4p7/F");
    tree->Branch("absdeltarapidity_hleadingjet_pt30_eta4p7_jesup",&absdeltarapidity_hleadingjet_pt30_eta4p7_jesup,"absdeltarapidity_hleadingjet_pt30_eta4p7_jesup/F");
    tree->Branch("absdeltarapidity_hleadingjet_pt30_eta4p7_jesdn",&absdeltarapidity_hleadingjet_pt30_eta4p7_jesdn,"absdeltarapidity_hleadingjet_pt30_eta4p7_jesdn/F");
    tree->Branch("absdeltarapidity_hleadingjet_pt30_eta4p7_jerup",&absdeltarapidity_hleadingjet_pt30_eta4p7_jerup,"absdeltarapidity_hleadingjet_pt30_eta4p7_jerup/F");
    tree->Branch("absdeltarapidity_hleadingjet_pt30_eta4p7_jerdn",&absdeltarapidity_hleadingjet_pt30_eta4p7_jerdn,"absdeltarapidity_hleadingjet_pt30_eta4p7_jerdn/F");
    tree->Branch("nbjets_pt30_eta4p7",&nbjets_pt30_eta4p7,"nbjets_pt30_eta4p7/I");
    tree->Branch("nvjets_pt40_eta2p4",&nvjets_pt40_eta2p4,"nvjets_pt40_eta2p4/I");
    tree->Branch("DijetMass",&DijetMass,"DijetMass/F");
    tree->Branch("DijetDEta",&DijetDEta,"DijetDEta/F");
    tree->Branch("DijetFisher",&DijetFisher,"DijetFisher/F");
    tree->Branch("njets_pt30_eta2p5",&njets_pt30_eta2p5,"njets_pt30_eta2p5/I");
    tree->Branch("njets_pt30_eta2p5_jesup",&njets_pt30_eta2p5_jesup,"njets_pt30_eta2p5_jesup/I");
    tree->Branch("njets_pt30_eta2p5_jesdn",&njets_pt30_eta2p5_jesdn,"njets_pt30_eta2p5_jesdn/I");
    tree->Branch("njets_pt30_eta2p5_jerup",&njets_pt30_eta2p5_jerup,"njets_pt30_eta2p5_jerup/I");
    tree->Branch("njets_pt30_eta2p5_jerdn",&njets_pt30_eta2p5_jerdn,"njets_pt30_eta2p5_jerdn/I");
    tree->Branch("pt_leadingjet_pt30_eta2p5",&pt_leadingjet_pt30_eta2p5,"pt_leadingjet_pt30_eta2p5/F");
    tree->Branch("pt_leadingjet_pt30_eta2p5_jesup",&pt_leadingjet_pt30_eta2p5_jesup,"pt_leadingjet_pt30_eta2p5_jesup/F");
    tree->Branch("pt_leadingjet_pt30_eta2p5_jesdn",&pt_leadingjet_pt30_eta2p5_jesdn,"pt_leadingjet_pt30_eta2p5_jesdn/F");
    tree->Branch("pt_leadingjet_pt30_eta2p5_jerup",&pt_leadingjet_pt30_eta2p5_jerup,"pt_leadingjet_pt30_eta2p5_jerup/F");
    tree->Branch("pt_leadingjet_pt30_eta2p5_jerdn",&pt_leadingjet_pt30_eta2p5_jerdn,"pt_leadingjet_pt30_eta2p5_jerdn/F");*/


    // merged jets
    //tree->Branch("mergedjet_iscleanH4l",&mergedjet_iscleanH4l);
    tree->Branch("mergedjet_pt",&mergedjet_pt);
    tree->Branch("mergedjet_eta",&mergedjet_eta);
    tree->Branch("mergedjet_phi",&mergedjet_phi);
    /*tree->Branch("mergedjet_mass",&mergedjet_mass);    
    tree->Branch("mergedjet_tau1",&mergedjet_tau1);
    tree->Branch("mergedjet_tau2",&mergedjet_tau2);
    tree->Branch("mergedjet_btag",&mergedjet_btag);
    
    tree->Branch("mergedjet_L1",&mergedjet_L1);
    tree->Branch("mergedjet_softdropmass",&mergedjet_softdropmass);
    tree->Branch("mergedjet_prunedmass",&mergedjet_prunedmass);

    tree->Branch("mergedjet_nsubjet",&mergedjet_nsubjet);
    tree->Branch("mergedjet_subjet_pt",&mergedjet_subjet_pt);
    tree->Branch("mergedjet_subjet_eta",&mergedjet_subjet_eta);
    tree->Branch("mergedjet_subjet_phi",&mergedjet_subjet_phi);
    tree->Branch("mergedjet_subjet_mass",&mergedjet_subjet_mass);
    tree->Branch("mergedjet_subjet_btag",&mergedjet_subjet_btag);
    tree->Branch("mergedjet_subjet_partonFlavour",&mergedjet_subjet_partonFlavour);
    tree->Branch("mergedjet_subjet_hadronFlavour",&mergedjet_subjet_hadronFlavour);*/

    // FSR Photons
    /*tree->Branch("nFSRPhotons",&nFSRPhotons,"nFSRPhotons/I");
    tree->Branch("allfsrPhotons_dR",&allfsrPhotons_dR);
    tree->Branch("allfsrPhotons_iso",&allfsrPhotons_iso);
    tree->Branch("allfsrPhotons_pt",&allfsrPhotons_pt);
    tree->Branch("fsrPhotons_lepindex",&fsrPhotons_lepindex);
    tree->Branch("fsrPhotons_pt",&fsrPhotons_pt_float);
    tree->Branch("fsrPhotons_pterr",&fsrPhotons_pterr_float);
    tree->Branch("fsrPhotons_eta",&fsrPhotons_eta_float);
    tree->Branch("fsrPhotons_phi",&fsrPhotons_phi_float);
    tree->Branch("fsrPhotons_dR",&fsrPhotons_dR);
    tree->Branch("fsrPhotons_iso",&fsrPhotons_iso);

    // Z4l? FIXME
    tree->Branch("theta12",&theta12,"theta12/F"); 
    tree->Branch("theta13",&theta13,"theta13/F"); 
    tree->Branch("theta14",&theta14,"theta14/F");
    tree->Branch("minM3l",&minM3l,"minM3l/F"); 
    tree->Branch("Z4lmaxP",&Z4lmaxP,"Z4lmaxP/F"); 
    tree->Branch("minDeltR",&minDeltR,"minDeltR/F"); 
    tree->Branch("m3l_soft",&m3l_soft,"m3l_soft/F");
    tree->Branch("minMass2Lep",&minMass2Lep,"minMass2Lep/F"); 
    tree->Branch("maxMass2Lep",&maxMass2Lep,"maxMass2Lep/F");
    tree->Branch("thetaPhoton",&thetaPhoton,"thetaPhoton/F"); 
    tree->Branch("thetaPhotonZ",&thetaPhotonZ,"thetaPhotonZ/F");*/

    // Event Category
    //tree->Branch("EventCat",&EventCat,"EventCat/I");

    // -------------------------                                                                                                                                                                        
    // GEN level information                                                                                                                                                                            
    // -------------------------                                                                                                                                                                        
    //Event variables
    //tree->Branch("GENfinalState",&GENfinalState,"GENfinalState/I");
    //tree->Branch("passedFiducialSelection",&passedFiducialSelection,"passedFiducialSelection/O");

    // lepton variables
    /*tree->Branch("GENlep_pt",&GENlep_pt_float);
    tree->Branch("GENlep_eta",&GENlep_eta_float);
    tree->Branch("GENlep_phi",&GENlep_phi_float);
    tree->Branch("GENlep_mass",&GENlep_mass_float);
    tree->Branch("GENlep_id",&GENlep_id);
    tree->Branch("GENlep_status",&GENlep_status);
    tree->Branch("GENlep_MomId",&GENlep_MomId);
    tree->Branch("GENlep_MomMomId",&GENlep_MomMomId);
    tree->Branch("GENlep_Hindex",&GENlep_Hindex,"GENlep_Hindex[4]/I");
    tree->Branch("GENlep_isoCH",&GENlep_isoCH);
    tree->Branch("GENlep_isoNH",&GENlep_isoNH);
    tree->Branch("GENlep_isoPhot",&GENlep_isoPhot);
    tree->Branch("GENlep_RelIso",&GENlep_RelIso);

    // Higgs candidate variables (calculated using selected gen leptons)
    tree->Branch("GENH_pt",&GENH_pt_float);
    tree->Branch("GENH_eta",&GENH_eta_float);
    tree->Branch("GENH_phi",&GENH_phi_float);
    tree->Branch("GENH_mass",&GENH_mass_float);
    tree->Branch("GENmass4l",&GENmass4l,"GENmass4l/F");
    tree->Branch("GENmass4mu",&GENmass4mu,"GENmass4mu/F");
    tree->Branch("GENmass4e",&GENmass4e,"GENmass4e/F");
    tree->Branch("GENmass2e2mu",&GENmass2e2mu,"GENmass2e2mu/F");
    tree->Branch("GENpT4l",&GENpT4l,"GENpT4l/F");
    tree->Branch("GENeta4l",&GENeta4l,"GENeta4l/F");
    tree->Branch("GENrapidity4l",&GENrapidity4l,"GENrapidity4l/F");
    

    // Z candidate variables
    tree->Branch("GENZ_pt",&GENZ_pt_float);
    tree->Branch("GENZ_eta",&GENZ_eta_float);
    tree->Branch("GENZ_phi",&GENZ_phi_float);
    tree->Branch("GENZ_mass",&GENZ_mass_float);
    tree->Branch("GENZ_DaughtersId",&GENZ_DaughtersId); 
    tree->Branch("GENZ_MomId",&GENZ_MomId);
    tree->Branch("GENmassZ1",&GENmassZ1,"GENmassZ1/F");
    tree->Branch("GENmassZ2",&GENmassZ2,"GENmassZ2/F");  
    tree->Branch("GENpTZ1",&GENpTZ1,"GENpTZ1/F");
    tree->Branch("GENpTZ2",&GENpTZ2,"GENpTZ2/F");
    tree->Branch("GENdPhiZZ",&GENdPhiZZ,"GENdPhiZZ/F");
    tree->Branch("GENmassZZ",&GENmassZZ,"GENmassZZ/F");
    tree->Branch("GENpTZZ",&GENpTZZ,"GENpTZZ/F");

    // Higgs variables directly from GEN particle
    tree->Branch("GENHmass",&GENHmass,"GENHmass/F");*/


    // Jets
    tree->Branch("GENjet_pt",&GENjet_pt_float);
    tree->Branch("GENjet_eta",&GENjet_eta_float);
    tree->Branch("GENjet_phi",&GENjet_phi_float);
    /*tree->Branch("GENjet_mass",&GENjet_mass_float);
    tree->Branch("GENnjets_pt30_eta4p7",&GENnjets_pt30_eta4p7,"GENnjets_pt30_eta4p7/I");
    tree->Branch("GENpt_leadingjet_pt30_eta4p7",&GENpt_leadingjet_pt30_eta4p7,"GENpt_leadingjet_pt30_eta4p7/F");
    tree->Branch("GENabsrapidity_leadingjet_pt30_eta4p7",&GENabsrapidity_leadingjet_pt30_eta4p7,"GENabsrapidity_leadingjet_pt30_eta4p7/F");
    tree->Branch("GENabsdeltarapidity_hleadingjet_pt30_eta4p7",&GENabsdeltarapidity_hleadingjet_pt30_eta4p7,"GENabsdeltarapidity_hleadingjet_pt30_eta4p7/F");
    tree->Branch("GENnjets_pt30_eta2p5",&GENnjets_pt30_eta2p5,"GENnjets_pt30_eta2p5/I");
    tree->Branch("GENpt_leadingjet_pt30_eta2p5",&GENpt_leadingjet_pt30_eta2p5,"GENpt_leadingjet_pt30_eta2p5/F");
    tree->Branch("lheNj",&lheNj,"lheNj/I");
    tree->Branch("lheNb",&lheNb,"lheNb/I");
    tree->Branch("nGenStatus2bHad",&nGenStatus2bHad,"nGenStatus2bHad/I");*/

    



}


/*void UFHZZ4LAna::setTreeVariables( const edm::Event& iEvent, const edm::EventSetup& iSetup,
                                   //std::vector<pat::Muon> selectedMuons, std::vector<pat::Electron> selectedElectrons, 
                                   //std::vector<pat::Muon> recoMuons, std::vector<pat::Electron> recoElectrons, 
                                   std::vector<pat::Jet> goodJets, std::vector<float> goodJetQGTagger, 
                                   std::vector<float> goodJetaxis2, std::vector<float> goodJetptD, std::vector<int> goodJetmult,
                                   std::vector<pat::Jet> selectedMergedJets
                                   //std::map<unsigned int, TLorentzVector> selectedFsrMap
                                   )*/
void UFHZZ4LAna::setTreeVariables( const edm::Event& iEvent, const edm::EventSetup& iSetup,
                                   std::vector<pat::Jet> goodJets, std::vector<float> goodJetQGTagger, 
                                   std::vector<float> goodJetaxis2, std::vector<float> goodJetptD, std::vector<int> goodJetmult,
                                   std::vector<pat::Jet> selectedMergedJets
                                   )
{

//    std::string DATAPATH = std::getenv( "CMSSW_BASE" );
//    if(year == 2018)    DATAPATH+="/src/UFHZZAnalysisRun2/KalmanMuonCalibrationsProducer/data/roccor.Run2.v3/RoccoR2018.txt";
//    if(year == 2017)    DATAPATH+="/src/UFHZZAnalysisRun2/KalmanMuonCalibrationsProducer/data/roccor.Run2.v3/RoccoR2017.txt";
//    if(year == 2016)    DATAPATH+="/src/UFHZZAnalysisRun2/KalmanMuonCalibrationsProducer/data/roccor.Run2.v3/RoccoR2016.txt";
//    RoccoR  *calibrator = new RoccoR(DATAPATH); 
//    calibrator = new RoccoR(DATAPATH);
   

    using namespace edm;
    using namespace pat;
    using namespace std;

    // Jet Info
    double tempDeltaR = 999.0;
    //std::cout<<"ELISA = "<<"good jets "<<goodJets.size()<<std::endl;
    for( unsigned int k = 0; k < goodJets.size(); k++) {

        jet_pt.push_back(goodJets[k].pt());
        jet_pt_raw.push_back(goodJets[k].pt());///jet Pt without JEC applied
        jet_eta.push_back(goodJets[k].eta());
        jet_phi.push_back(goodJets[k].phi());
        //jet_mass.push_back(goodJets[k].M());

       
    } // loop over jets



    // Higgs Variables
    /*if( RecoFourMuEvent ){ finalState = 1;}
    if( RecoFourEEvent  ){ finalState = 2;}
    if( RecoTwoETwoMuEvent ){ finalState = 3;}
    if( RecoTwoMuTwoEEvent ){ finalState = 4;}

    H_pt.push_back(HVec.Pt());
    H_eta.push_back(HVec.Eta());
    H_phi.push_back(HVec.Phi());
    H_mass.push_back(HVec.M());
    mass4l = HVec.M();
    mass4l_noFSR = HVecNoFSR.M();

    if(RecoFourMuEvent){mass4mu = HVec.M();}
    else{mass4mu = -1;}
    if(RecoFourEEvent){ mass4e = HVec.M();}
    else{ mass4e = -1;}
    if(RecoTwoETwoMuEvent || RecoTwoMuTwoEEvent){ mass2e2mu = HVec.M(); }
    else{ mass2e2mu = -1;}

    pT4l = HVec.Pt(); eta4l = HVec.Eta(); rapidity4l = HVec.Rapidity(); phi4l = HVec.Phi();

    pTZ1 = Z1Vec.Pt(); pTZ2 = Z2Vec.Pt(); massZ1 = Z1Vec.M(); massZ2 = Z2Vec.M();
	
    if (njets_pt30_eta4p7>0) absdeltarapidity_hleadingjet_pt30_eta4p7 = fabs(rapidity4l-absrapidity_leadingjet_pt30_eta4p7);
    if (njets_pt30_eta4p7_jesup>0) absdeltarapidity_hleadingjet_pt30_eta4p7_jesup = fabs(rapidity4l-absrapidity_leadingjet_pt30_eta4p7_jesup);
    if (njets_pt30_eta4p7_jesdn>0) absdeltarapidity_hleadingjet_pt30_eta4p7_jesdn = fabs(rapidity4l-absrapidity_leadingjet_pt30_eta4p7_jesdn);
    if (njets_pt30_eta4p7_jerup>0) absdeltarapidity_hleadingjet_pt30_eta4p7_jerup = fabs(rapidity4l-absrapidity_leadingjet_pt30_eta4p7_jerup);
    if (njets_pt30_eta4p7_jerdn>0) absdeltarapidity_hleadingjet_pt30_eta4p7_jerdn = fabs(rapidity4l-absrapidity_leadingjet_pt30_eta4p7_jerdn);
    
    if (njets_pt30_eta4p7>0) absrapidity_leadingjet_pt30_eta4p7 = fabs(absrapidity_leadingjet_pt30_eta4p7);
    if (njets_pt30_eta4p7_jesup>0) absrapidity_leadingjet_pt30_eta4p7_jesup = fabs(absrapidity_leadingjet_pt30_eta4p7_jesup);
    if (njets_pt30_eta4p7_jesdn>0) absrapidity_leadingjet_pt30_eta4p7_jesdn = fabs(absrapidity_leadingjet_pt30_eta4p7_jesdn);
    if (njets_pt30_eta4p7_jerup>0) absrapidity_leadingjet_pt30_eta4p7_jerup = fabs(absrapidity_leadingjet_pt30_eta4p7_jerup);
    if (njets_pt30_eta4p7_jerdn>0) absrapidity_leadingjet_pt30_eta4p7_jerdn = fabs(absrapidity_leadingjet_pt30_eta4p7_jerdn);

    //std::cout<<"Higgs = "<<foundHiggsCandidate<<std::endl;
    
    if (foundHiggsCandidate) {
	    //std::cout<<"finalState = "<<finalState<<std::endl;

        TLorentzVector Lep1FSR, Lep2FSR, Lep3FSR, Lep4FSR;
        Lep1FSR.SetPtEtaPhiM(lepFSR_pt[lep_Hindex[0]],lepFSR_eta[lep_Hindex[0]],lepFSR_phi[lep_Hindex[0]],lepFSR_mass[lep_Hindex[0]]);
        Lep2FSR.SetPtEtaPhiM(lepFSR_pt[lep_Hindex[1]],lepFSR_eta[lep_Hindex[1]],lepFSR_phi[lep_Hindex[1]],lepFSR_mass[lep_Hindex[1]]);
        Lep3FSR.SetPtEtaPhiM(lepFSR_pt[lep_Hindex[2]],lepFSR_eta[lep_Hindex[2]],lepFSR_phi[lep_Hindex[2]],lepFSR_mass[lep_Hindex[2]]);
        Lep4FSR.SetPtEtaPhiM(lepFSR_pt[lep_Hindex[3]],lepFSR_eta[lep_Hindex[3]],lepFSR_phi[lep_Hindex[3]],lepFSR_mass[lep_Hindex[3]]);
        
        pTL1FSR = Lep1FSR.Pt(); etaL1FSR = Lep1FSR.Eta(); phiL1FSR = Lep1FSR.Phi();	mL1FSR = Lep1FSR.M();
        pTL2FSR = Lep2FSR.Pt(); etaL2FSR = Lep2FSR.Eta(); phiL2FSR = Lep2FSR.Phi();	mL2FSR = Lep2FSR.M();
        pTL3FSR = Lep3FSR.Pt(); etaL3FSR = Lep3FSR.Eta(); phiL3FSR = Lep3FSR.Phi();	mL3FSR = Lep3FSR.M();
        pTL4FSR = Lep4FSR.Pt(); etaL4FSR = Lep4FSR.Eta(); phiL4FSR = Lep4FSR.Phi();	mL4FSR = Lep4FSR.M();

        TLorentzVector Lep1, Lep2, Lep3, Lep4;
        Lep1.SetPtEtaPhiM(lep_pt[lep_Hindex[0]],lep_eta[lep_Hindex[0]],lep_phi[lep_Hindex[0]],lep_mass[lep_Hindex[0]]);
        Lep2.SetPtEtaPhiM(lep_pt[lep_Hindex[1]],lep_eta[lep_Hindex[1]],lep_phi[lep_Hindex[1]],lep_mass[lep_Hindex[1]]);
        Lep3.SetPtEtaPhiM(lep_pt[lep_Hindex[2]],lep_eta[lep_Hindex[2]],lep_phi[lep_Hindex[2]],lep_mass[lep_Hindex[2]]);
        Lep4.SetPtEtaPhiM(lep_pt[lep_Hindex[3]],lep_eta[lep_Hindex[3]],lep_phi[lep_Hindex[3]],lep_mass[lep_Hindex[3]]);
        idL1 = lep_id[lep_Hindex[0]]; pTL1 = Lep1.Pt(); etaL1 = Lep1.Eta(); pTErrL1 = lep_pterr[lep_Hindex[0]]; mL1 = lep_mass[lep_Hindex[0]]; phiL1 = lep_phi[lep_Hindex[0]];
        idL2 = lep_id[lep_Hindex[1]]; pTL2 = Lep2.Pt(); etaL2 = Lep2.Eta(); pTErrL2 = lep_pterr[lep_Hindex[1]]; mL2 = lep_mass[lep_Hindex[1]]; phiL2 = lep_phi[lep_Hindex[1]];
        idL3 = lep_id[lep_Hindex[2]]; pTL3 = Lep3.Pt(); etaL3 = Lep3.Eta(); pTErrL3 = lep_pterr[lep_Hindex[2]]; mL3 = lep_mass[lep_Hindex[2]]; phiL3 = lep_phi[lep_Hindex[2]];
        idL4 = lep_id[lep_Hindex[3]]; pTL4 = Lep4.Pt(); etaL4 = Lep4.Eta(); pTErrL4 = lep_pterr[lep_Hindex[3]]; mL4 = lep_mass[lep_Hindex[3]]; phiL4 = lep_phi[lep_Hindex[3]];
     

    }*/


}


/*void UFHZZ4LAna::setGENVariables(//edm::Handle<reco::GenParticleCollection> prunedgenParticles,
                                 //edm::Handle<edm::View<pat::PackedGenParticle> > packedgenParticles,
                                 edm::Handle<edm::View<reco::GenJet> > genJets)*/
void UFHZZ4LAna::setGENVariables(edm::Handle<edm::View<reco::GenJet> > genJets)
{

        edm::View<reco::GenJet>::const_iterator genjet; 
	for(genjet = genJets->begin(); genjet != genJets->end(); genjet++) {
		                          
		  double pt = genjet->pt();  double eta = genjet->eta();
		  //if (pt<30.0 || abs(eta)>4.7) continue;

		  
      GENjet_pt.push_back(genjet->pt());
      GENjet_eta.push_back(genjet->eta());
      GENjet_phi.push_back(genjet->phi());
      GENjet_mass.push_back(genjet->mass());

	}// loop over gen jets

         
}

/*bool UFHZZ4LAna::mZ1_mZ2(unsigned int& L1, unsigned int& L2, unsigned int& L3, unsigned int& L4, bool makeCuts)
{

    double offshell = 999.0; bool findZ1 = false; bool passZ1 = false;

    L1 = 0; L2 = 0;

    unsigned int N = GENlep_pt.size();

    for(unsigned int i=0; i<N; i++){
        for(unsigned int j=i+1; j<N; j++){


            if((GENlep_id[i]+GENlep_id[j])!=0) continue;

            TLorentzVector li, lj;
            li.SetPtEtaPhiM(GENlep_pt[i],GENlep_eta[i],GENlep_phi[i],GENlep_mass[i]);
            lj.SetPtEtaPhiM(GENlep_pt[j],GENlep_eta[j],GENlep_phi[j],GENlep_mass[j]);

            if (verbose) cout<<"gen lep i id: "<<GENlep_id[i]<<" pt: "<<li.Pt()<<" lep j id: "<<GENlep_id[j]<<" pt: "<<lj.Pt()<<endl;

            if (makeCuts) {
                if ( abs(GENlep_id[i]) == 13 && (li.Pt() < 5.0 || abs(li.Eta()) > 2.4)) continue;
                if ( abs(GENlep_id[i]) == 11 && (li.Pt() < 7.0 || abs(li.Eta()) > 2.5)) continue;
                if ( GENlep_RelIso[i]>((abs(GENlep_id[i])==11)?genIsoCutEl:genIsoCutMu)) continue;
                
                if ( abs(GENlep_id[j]) == 13 && (lj.Pt() < 5.0 || abs(lj.Eta()) > 2.4)) continue;
                if ( abs(GENlep_id[j]) == 11 && (lj.Pt() < 7.0 || abs(lj.Eta()) > 2.5)) continue;
                if ( GENlep_RelIso[j]>((abs(GENlep_id[i])==11)?genIsoCutEl:genIsoCutMu)) continue;                
            }

            TLorentzVector mll = li+lj;
            if (verbose) cout<<"gen mass ij: "<<mll.M()<<endl;

            if(abs(mll.M()-Zmass)<offshell){
                double mZ1 = mll.M();
                if (verbose) cout<<"foundZ1"<<endl;
                L1 = i; L2 = j; findZ1 = true; offshell = abs(mZ1-Zmass);          
            }
        }    
    }

    TLorentzVector l1, l2;
    l1.SetPtEtaPhiM(GENlep_pt[L1],GENlep_eta[L1],GENlep_phi[L1],GENlep_mass[L1]);
    l2.SetPtEtaPhiM(GENlep_pt[L2],GENlep_eta[L2],GENlep_phi[L2],GENlep_mass[L2]);
    TLorentzVector ml1l2 = l1+l2;

    if(ml1l2.M()>40 && ml1l2.M()<120 && findZ1) passZ1 = true;
    if (!makeCuts) passZ1 = true;

    double pTL34 = 0.0; bool findZ2 = false; 
    //bool m4lwindow=false; double window_lo=70.0; double window_hi=140.0;
   
    //cout<<"findZ2"<<endl;
    for(unsigned int i=0; i<N; i++){
        if(i==L1 || i==L2) continue; // can not be the lep from Z1
        for(unsigned int j=i+1; j<N; j++){
            if(j==L1 || j==L2) continue; // can not be the lep from Z1
            if((GENlep_id[i]+GENlep_id[j])!=0) continue;            

            TLorentzVector li, lj;
            li.SetPtEtaPhiM(GENlep_pt[i],GENlep_eta[i],GENlep_phi[i],GENlep_mass[i]);
            lj.SetPtEtaPhiM(GENlep_pt[j],GENlep_eta[j],GENlep_phi[j],GENlep_mass[j]);
            TLorentzVector Z2 = li+lj;

            if (makeCuts) {
                if ( abs(GENlep_id[i]) == 13 && (li.Pt() < 5.0 || abs(li.Eta()) > 2.4)) continue;
                if ( abs(GENlep_id[i]) == 11 && (li.Pt() < 7.0 || abs(li.Eta()) > 2.5)) continue;
                if ( GENlep_RelIso[i]>((abs(GENlep_id[i])==11)?genIsoCutEl:genIsoCutMu)) continue;
                
                if ( abs(GENlep_id[j]) == 13 && (lj.Pt() < 5.0 || abs(lj.Eta()) > 2.4)) continue;
                if ( abs(GENlep_id[j]) == 11 && (lj.Pt() < 7.0 || abs(lj.Eta()) > 2.5)) continue;
                if ( GENlep_RelIso[j]>((abs(GENlep_id[i])==11)?genIsoCutEl:genIsoCutMu)) continue;
            }

            if ( (li.Pt()+lj.Pt())>=pTL34 ) {
                double mZ2 = Z2.M();
                if (verbose) cout<<"GEN mZ2: "<<mZ2<<endl;
                if( (mZ2>12 && mZ2<120) || (!makeCuts) ) {
                    L3 = i; L4 = j; findZ2 = true; 
                    pTL34 = li.Pt()+lj.Pt();
                    if (verbose) cout<<"is the new GEN cand"<<endl;
                    //if (m4l>window_lo && m4l<window_hi) m4lwindow=true;
                } else {
                    // still assign L3 and L4 to this pair if we don't have a passing Z2 yet
                    if (findZ2 == false) {L3 = i; L4 = j;}
                    //cout<<"is not new GEN cand"<<endl;
                }
            }
            
        } // lj
    } // li

    if(passZ1 && findZ2) return true;
    else return false;
    
}*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
UFHZZ4LAna::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(UFHZZ4LAna);

//  LocalWords:  ecalDriven
  


