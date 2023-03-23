////////////////////////////////////////////////////////////////////////
// Class:       pi0TestSelection
// Plugin Type: analyzer (art v2_07_03)
// File:        pi0TestSelection_module.cc
//TODO follow new calibration as done in PDSPAnalyser (after understanding current reconsturction)
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "larsim/MCCheater/BackTrackerService.h"
#include "larsim/MCCheater/ParticleInventoryService.h"
#include "lardata/DetectorInfoServices/DetectorClocksService.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesService.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/RecoBase/Shower.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "nusimdata/SimulationBase/MCTruth.h"
#include "lardataobj/AnalysisBase/CosmicTag.h"
#include "lardataobj/AnalysisBase/T0.h"
#include "lardataobj/AnalysisBase/Calorimetry.h"
#include "lardataobj/RecoBase/OpFlash.h"
#include "larcore/Geometry/Geometry.h"
#include "larreco/RecoAlg/TrackMomentumCalculator.h"

//protoDUNE analysis headers
#include "protoduneana/Utilities/ProtoDUNETrackUtils.h"
#include "protoduneana/Utilities/ProtoDUNEShowerUtils.h"
#include "protoduneana/Utilities/ProtoDUNETruthUtils.h"
#include "protoduneana/Utilities/ProtoDUNEPFParticleUtils.h"
#include "protoduneana/Utilities/ProtoDUNEBeamlineUtils.h"
#include "protoduneana/Utilities/ProtoDUNECalibration.h"

//dunetpc headers
#include "dunecore/DuneObj/ProtoDUNEBeamEvent.h"

//ROOT includes
#include <TTree.h>
#include <TH1D.h>
#include "Math/Vector3D.h"

//Use this to get CNN output
#include "lardata/ArtDataHelper/MVAReader.h"

#include <algorithm>
#include <chrono>

namespace protoana {
  class pi0TestSelection;
}


class protoana::pi0TestSelection : public art::EDAnalyzer {
  public:

  explicit pi0TestSelection(fhicl::ParameterSet const & p);
  // The compiler-generated destructor is fine for non-base
  // classes without bare pointers or other resource use.

  // Plugins should not be copied or assigned.
  pi0TestSelection(pi0TestSelection const &) = delete;
  pi0TestSelection(pi0TestSelection &&) = delete;
  pi0TestSelection & operator = (pi0TestSelection const &) = delete;
  pi0TestSelection & operator = (pi0TestSelection &&) = delete;

  virtual void beginJob() override;
  virtual void endJob() override;

  // Required functions.
  void analyze(art::Event const & e) override;

  // Custom functions.
  std::vector<art::Ptr<recob::Hit>> GetMCParticleArtHits(const art::Event &evt, detinfo::DetectorClocksData const& clockData, const simb::MCParticle &mcpart, const std::vector<art::Ptr<recob::Hit> > &hitVec, bool use_eve = true) const;
  int PandoraIdentification(const recob::PFParticle &daughterPFP, const art::Event &evt);
  std::vector<double> CNNScoreCalculator(anab::MVAReader<recob::Hit,4> &hitResults, const std::vector< art::Ptr< recob::Hit > > &hits, unsigned int &n);
  std::vector<std::vector<double> > StartHits(unsigned int n_hits, art::FindManyP<recob::SpacePoint> spFromHits, TVector3 &showerStart, TVector3 &direction);
  double ShowerEnergyCalculator(const std::vector<art::Ptr<recob::Hit> > &hits, const detinfo::DetectorPropertiesData &detProp, art::FindManyP<recob::SpacePoint> &spFromHits);
  int CountHitsInPlane(std::vector<const recob::Hit *> &hits, int plane);
  double YPositionAssumption(const std::vector<art::Ptr<recob::Hit> > &hits, const detinfo::DetectorPropertiesData &detProp, art::FindManyP<recob::SpacePoint> &spFromHits);
  double HitEnergyCalculator(const art::Ptr<recob::Hit> &hit, const detinfo::DetectorPropertiesData &detProp, const double &y_assumption, const std::vector<art::Ptr<recob::SpacePoint>> &spacePoints);

  void reset();

  void AnalyseDaughterPFP(const recob::PFParticle &daughterPFP, const art::Event &evt, const detinfo::DetectorPropertiesData &detProp, anab::MVAReader<recob::Hit,4> &hitResults);
  void AnalyseBeamPFP(const recob::PFParticle &beam, const art::Event &evt);
  void AnalyseMCTruth(const recob::PFParticle &daughter, const art::Event &evt, const detinfo::DetectorPropertiesData &detProp, const detinfo::DetectorClocksData &clockData, const std::vector<art::Ptr<recob::Hit> > &hitVec);
  void AnalyseMCTruthBeam(const art::Event &evt);
  void FillG4NTuple(const sim::ParticleList &particle_list, const simb::MCParticle* &particle, const int &number);
  void CollectG4Particle(const int &Pdg, const int start, const int stop);
  void AnalyseFromBeam(const art::Event &evt, const detinfo::DetectorClocksData &clockData, const detinfo::DetectorPropertiesData &detProp, anab::MVAReader<recob::Hit,4> &hitResults, std::vector<recob::PFParticle> pfpVec);

  void NullRecoBeamInfo();
  void NullRecoDaughterPFPInfo();

  void CorrectBacktrackedHitEnergy();

  private:
  
  enum G4Mode{PI0=1, DIPHOTON=2, ALL=3, PI0_PIP=4, NONE=0}; // determines what MC particles are retrieved from the truth table
  
  // fcl parameters, order matters!
  protoana::ProtoDUNECalibration calibration_SCE;
  std::string fBeamCalorimetryTag;
  std::string fShowerCalorimetryTag;
  std::string fTrackerTag;
  std::string fShowerTag;
  std::string fHitTag;
  std::string fPFParticleTag;
  std::string fGeneratorTag;
  protoana::ProtoDUNEBeamlineUtils fBeamlineUtils;
  art::ServiceHandle<geo::Geometry> geom;
  art::ServiceHandle<cheat::BackTrackerService> bt_serv;
  art::ServiceHandle< cheat::ParticleInventoryService > pi_serv;

  bool fPi0Only;
  bool fRetrieveHitInfo;
  G4Mode fRetrieveG4;
  bool fDebug;

  // Initialise protodune analysis utility classes
  protoana::ProtoDUNEPFParticleUtils pfpUtil;
  protoana::ProtoDUNETrackUtils trkUtil;
  protoana::ProtoDUNETruthUtils truthUtil;
  protoana::ProtoDUNEShowerUtils showerUtil;
  protoana::ProtoDUNETrackUtils trackUtil;
  // local variables
  TTree *fOutTree = new TTree;

  // null arrays
  std::vector<double> null_double_array = {-999};
  std::vector<int> null_int_array = {-999};


  //---------------------------PFOs---------------------------//
  // PFO meta-data
  std::vector<int> PFPNum;
  std::vector<int> PFPMother;
  std::vector<int> sliceID;
  std::vector<int> beamCosmicScore;
  std::vector<int> pdgCodes; // pandora assumed pdg codes

  // track-shower identification
  std::vector<int> pandoraTags; // track/shower like tag from pandora
  std::vector<double> emScore;
  std::vector<double> trackScore;
  std::vector<double> CNNScore; // CNN score per shower

  // shower start position
  std::vector<double> startPosX;
  std::vector<double> startPosY;
  std::vector<double> startPosZ;

  // shower direction
  std::vector<double> dirX;
  std::vector<double> dirY;
  std::vector<double> dirZ;

  std::vector<double> coneAngle; // shower angle
  std::vector<double> length; // shower length

  // energy quantities
  std::vector<int> nHits; // number of hits on all planes
  std::vector<int> nHits_collection; // number of collection plane hits
  std::vector<double> energy; // reco shower energy in MeV
  std::vector<double> calibrated_energy; // reco shower calibrated energy in MeV?

  // hit quantities
  std::vector<std::vector<int>> hit_channel;
  std::vector<std::vector<double>> hit_peakTime;
  std::vector<std::vector<double>> hit_integral;
  
  std::vector<std::vector<double>> hit_energy;
  std::vector<std::vector<double>> spacePointX;
  std::vector<std::vector<double>> spacePointY;
  std::vector<std::vector<double>> spacePointZ;

  //---------------------------Reconstructed Beam---------------------------//
  int beam;
  int beamSliceID;

  // beam start position
  double beamStartPosX;
  double beamStartPosY;
  double beamStartPosZ;

  // beam end position
  double beamEndPosX;
  double beamEndPosY;
  double beamEndPosZ;

  double reco_beam_vertex_michel_score;
  int reco_beam_vertex_nHits;

  std::vector<double> reco_beam_calo_wire;
  std::vector<double> reco_beam_calibrated_dEdX_SCE;

  //---------------------------Backtracked Beam---------------------------//
  // true beam meta-data
  int trueBeamPdg;

  double trueBeamEnergy;
  double trueBeamMass;

  // true beam start positions
  double trueBeamStartPosX;
  double trueBeamStartPosY;
  double trueBeamStartPosZ;

  // true beam end positions
  double trueBeamEndPosX;
  double trueBeamEndPosY;
  double trueBeamEndPosZ;

  //---------------------------Backtracked Daughter---------------------------//
  std::vector<int> matchedNum;
  std::vector<int> matchedMother;
  std::vector<int> trueDaughterPDG;
  std::vector<int> matchedMotherPDG;

  // true daughter start positions
  std::vector<double> trueDaughterStartPosX;
  std::vector<double> trueDaughterStartPosY;
  std::vector<double> trueDaughterStartPosZ;

  // true daughter end positions
  std::vector<double> trueDaughterEndPosX;
  std::vector<double> trueDaughterEndPosY;
  std::vector<double> trueDaughterEndPosZ;

  // true daughter momentum
  std::vector<double> trueDaughterMomentumX;
  std::vector<double> trueDaughterMomentumY;
  std::vector<double> trueDaughterMomentumZ;

  std::vector<double> trueDaughterEnergy; // mc shower energy in MeV
  std::vector<double> trueDaughterMass;

  std::vector<int> sharedHits;
  std::vector<int> sharedHits_collection;
  std::vector<int> mcParticleHits;
  std::vector<int> mcParticleHits_collection;
  std::vector<int> hitsInRecoCluster;
  std::vector<int> hitsInRecoCluster_collection;
  std::vector<double> mcParticleEnergyByHits;
  std::vector<double> mcParticleEnergyByHits_shared_energy;

  std::vector<std::vector<int>> true_hit_channel;
  std::vector<std::vector<double>> true_hit_peakTime;
  std::vector<std::vector<double>> true_hit_integral;

  std::vector<std::vector<double>> true_hit_energy;
  std::vector<std::vector<double>> true_hit_spacePointX;
  std::vector<std::vector<double>> true_hit_spacePointY;
  std::vector<std::vector<double>> true_hit_spacePointZ;

  //---------------------------MC Truth---------------------------//
  std::vector<int> G4ParticleNum;
  std::vector<int> G4ParticleMother;
  std::vector<int> G4ParticleMotherPdg;
  std::vector<int> G4ParticlePdg;

  std::vector<double> G4ParticleEnergy;
  std::vector<double> G4ParticleMass;

  std::vector<double> G4ParticleStartPosX;
  std::vector<double> G4ParticleStartPosY;
  std::vector<double> G4ParticleStartPosZ;
  
  std::vector<double> G4ParticleEndPosX;
  std::vector<double> G4ParticleEndPosY;
  std::vector<double> G4ParticleEndPosZ;

  std::vector<double> G4ParticleMomX;
  std::vector<double> G4ParticleMomY;
  std::vector<double> G4ParticleMomZ;

  //---------------------------Event info---------------------------//
  unsigned int run;
  unsigned int subRun;
  unsigned int eventID;

  double totalEvents; // number of events processed
  double beamEvents; // number of events with beam particles
};


protoana::pi0TestSelection::pi0TestSelection(fhicl::ParameterSet const & p)
  :
  EDAnalyzer(p),
  calibration_SCE(p.get<fhicl::ParameterSet>("CalibrationParsSCE")),
  fBeamCalorimetryTag(p.get<std::string>("BeamCalorimetryTag")),
  fShowerCalorimetryTag(p.get<std::string>("ShowerCalorimetryTag")),
  fTrackerTag(p.get<std::string>("TrackerTag")),
  fShowerTag(p.get<std::string>("ShowerTag")),
  fHitTag(p.get<std::string>("HitTag")),
  fPFParticleTag(p.get<std::string>("PFParticleTag")),
  fGeneratorTag(p.get<std::string>("GeneratorTag")),
  fBeamlineUtils(p.get<fhicl::ParameterSet>("BeamlineUtils")),
  fPi0Only(p.get<bool>("Pi0Only")),
  fRetrieveHitInfo(p.get<bool>("RetrieveHitInfo")),
  fRetrieveG4(static_cast<G4Mode>(p.get<int>("RetrieveG4"))),
  fDebug(p.get<bool>("Debug"))
{ }

// shower energy calculation, taken from Jake Calcutt's PDPSPAnalyser
double protoana::pi0TestSelection::ShowerEnergyCalculator(const std::vector<art::Ptr<recob::Hit> > &hits, const detinfo::DetectorPropertiesData &detProp, art::FindManyP<recob::SpacePoint> &spFromHits)
{
  std::vector<double> x_vec, y_vec, z_vec; // parameterised hit position vector for each hit
  double total_y = 0;
  int n_good_y = 0;
  std::vector<art::Ptr<recob::Hit>> good_hits;
  for(unsigned int i = 0; i < hits.size(); i++)
  {
    auto hit = hits[i];
    // skip any hits not on the collection plane (shouldn't be anyways)
    if(hit->View() != 2)
    {
      continue;
    }

    good_hits.push_back(hit);

    double shower_hit_x = detProp.ConvertTicksToX(hit->PeakTime(), hit->WireID().Plane, hit->WireID().TPC, 0);
    double shower_hit_z = geom->Wire(hit->WireID()).GetCenter().Z();

    x_vec.push_back(shower_hit_x);
    z_vec.push_back(shower_hit_z);

    std::vector<art::Ptr<recob::SpacePoint>> sps = spFromHits.at(i);
    if (!sps.empty())
    {
      y_vec.push_back(sps[0]->XYZ()[1]);
      total_y += y_vec.back();
      n_good_y++;
    }
    else
    {
      y_vec.push_back(-999.);
    }
  }

  double total_energy = 0;
  if(n_good_y < 1)
  {
    if(fDebug) std::cout << "could not reconstruct energy" << std::endl;
    total_energy = -999;
  }
  else
  {
    for(unsigned int j = 0; j < good_hits.size(); j++)
    {

      auto good_hit = good_hits[j];
      
      if(good_hit->View() != 2)
      {
        continue;
      }

      if (y_vec[j] < -100.)
      {
        y_vec[j] = total_y / n_good_y;
      }
      total_energy += calibration_SCE.HitToEnergy(good_hit, x_vec[j], y_vec[j], z_vec[j]);
    }
  }
  return total_energy;
}


double protoana::pi0TestSelection::YPositionAssumption(const std::vector<art::Ptr<recob::Hit> > &hits, const detinfo::DetectorPropertiesData &detProp, art::FindManyP<recob::SpacePoint> &spFromHits)
{
  double total_y = 0;
  int n_good_y = 0;
  std::vector<art::Ptr<recob::Hit>> good_hits;
  for(unsigned int i = 0; i < hits.size(); i++)
  {
    auto hit = hits[i];
    // skip any hits not on the collection plane (shouldn't be anyways)
    if(hit->View() != 2){continue;}

    good_hits.push_back(hit);

    std::vector<art::Ptr<recob::SpacePoint>> sps = spFromHits.at(i);
    if (!sps.empty())
    {
      total_y += sps[0]->XYZ()[1];
      n_good_y++;
    }
  }

  return total_y / n_good_y;
}


double protoana::pi0TestSelection::HitEnergyCalculator(const art::Ptr<recob::Hit> &hit, const detinfo::DetectorPropertiesData &detProp, const double& y_assumption, const std::vector<art::Ptr<recob::SpacePoint>>& spacePoints)
{
  if(hit -> View() != 2){return -999;} // skip induction planes

  // make an assumption on paramaterized y position if no space points exist
  double y_param;
  if(!spacePoints.empty())
  {
    y_param = spacePoints[0]->XYZ()[1];
  }
  else
  {
    y_param = y_assumption;
  }

  TVector3 position(
    detProp.ConvertTicksToX(hit->PeakTime(), hit->WireID().Plane, hit->WireID().TPC, 0),
    y_param,
    geom->Wire(hit->WireID()).GetCenter().Z()
  ); // paramaterized positions

  double energy = calibration_SCE.HitToEnergy(hit, position.X(), position.Y(), position.Z());
  if(hit->PeakTime() == 4451.28857421875 && hit->Channel() == 2247)
  {
    std::cout << "hit peak channel: " << hit->Channel() << std::endl;
    std::cout << "hit peak peak time: " << hit->PeakTime() << std::endl;
    std::cout << "paramaterised position x: " << position.X() << std::endl;
    std::cout << "paramaterised position y: " << position.Y() << std::endl;
    std::cout << "paramaterised position z: " << position.Z() << std::endl;
    std::cout << "energy: " << energy << std::endl;
  }


  return energy;
}


std::vector<art::Ptr<recob::Hit> > protoana::pi0TestSelection::GetMCParticleArtHits(const art::Event &evt, detinfo::DetectorClocksData const& clockData, const simb::MCParticle &mcpart, const std::vector<art::Ptr<recob::Hit> > &hitVec, bool use_eve) const
{
  // Backtrack all hits to verify whether they belong to the current MCParticle.
  std::vector<art::Ptr<recob::Hit> > outVec;
  for(const art::Ptr<recob::Hit> hit : hitVec)
  {
    if (use_eve) {
      for(const sim::TrackIDE & ide : bt_serv->HitToEveTrackIDEs(clockData, *hit.get())) {
        int trackId = ide.trackID;
        if(pi_serv->TrackIdToParticle_P(trackId) == 
           pi_serv->TrackIdToParticle_P(mcpart.TrackId())) {
          outVec.push_back(hit);
          break;
        }
      }
    }
    else {
      for(const int trackId : bt_serv->HitToTrackIds(clockData, *hit.get())) {
        if(pi_serv->TrackIdToParticle_P(trackId) == 
           pi_serv->TrackIdToParticle_P(mcpart.TrackId())) {
          outVec.push_back(hit);
          break;
        }
      }
    }
  }
  return outVec;
}


// track/shower identification done thorugh pandora, returns 11 for a shower and 13 for a track
int protoana::pi0TestSelection::PandoraIdentification(const recob::PFParticle &daughterPFP, const art::Event &evt)
{
    // determine if they are track like or shower like using pandora
    // then fill a vector containing this data: 11 = shower 13 = track
    if(pfpUtil.IsPFParticleShowerlike(daughterPFP, evt, fPFParticleTag, fShowerTag))
    {
      return 11;
    }
    else if(pfpUtil.IsPFParticleTracklike(daughterPFP, evt, fPFParticleTag, fTrackerTag))
    {
      return 13;
    }
    else
    {
      return -999;
    }
    return -1;
}

// Calculates the CNN score of the event. Used to determine if a an event is track or shower like.
std::vector<double> protoana::pi0TestSelection::CNNScoreCalculator(anab::MVAReader<recob::Hit,4> &hitResults, const std::vector< art::Ptr< recob::Hit > > &hits, unsigned int &n)
{
  std::vector<double> output{};
  double score = 0;
  double mean_em = 0;
  double mean_track = 0;
  double cnn_track = 0;
  double cnn_em = 0;

  // Calculate the score per hit than take the average
  for(unsigned int h = 0; h < n; h++)
  {
    std::array<float,4> cnn_out = hitResults.getOutput( hits[h] );
    cnn_track = cnn_out[ hitResults.getIndex("track") ];
    cnn_em = cnn_out[ hitResults.getIndex("em") ];

    mean_em += cnn_em;
    mean_track += cnn_track;
  }
  mean_em = (n > 0) ? ( mean_em / n ) : -999; // posts -999 if there were no hits
  mean_track = (n > 0) ? ( mean_track / n ) : -999;
  score = (n > 0) ? (mean_em / (mean_em + mean_track) ) : -999;

  output.push_back(score);
  output.push_back(mean_em);
  output.push_back(mean_track);

  return output;
}


int protoana::pi0TestSelection::CountHitsInPlane(std::vector<const recob::Hit *> &hits, int plane)
{
  int n = 0;
  for(size_t i = 0; i < hits.size(); i++)
  {
    if(hits[i]->View() == plane)
    {
      n++;
    }
  }
  return n;
}


// Clears the various analyser outputs at the start of a new event to remove the previous events contents
void protoana::pi0TestSelection::reset()
{
  //---------------------------PFOs---------------------------//
  PFPNum.clear();
  PFPMother.clear();
  sliceID.clear();
  beamCosmicScore.clear();
  pdgCodes.clear();

  pandoraTags.clear();
  emScore.clear();
  trackScore.clear();
  CNNScore.clear();

  startPosX.clear();
  startPosY.clear();
  startPosZ.clear();

  dirX.clear();
  dirY.clear();
  dirZ.clear();

  coneAngle.clear();
  length.clear();

  nHits.clear();
  nHits_collection.clear();
  energy.clear();
  calibrated_energy.clear();

  hit_channel.clear();
  hit_peakTime.clear();
  hit_integral.clear();

  hit_energy.clear();
  spacePointX.clear();
  spacePointY.clear();
  spacePointZ.clear();

  //---------------------------Reconstructed Beam---------------------------//
  reco_beam_calo_wire.clear();
  reco_beam_calibrated_dEdX_SCE.clear();

  //---------------------------Backtracked Daughter---------------------------//
  matchedNum.clear();
  matchedMother.clear();
  trueDaughterPDG.clear();
  matchedMotherPDG.clear();

  trueDaughterStartPosX.clear();
  trueDaughterStartPosY.clear();
  trueDaughterStartPosZ.clear();

  trueDaughterEndPosX.clear();
  trueDaughterEndPosY.clear();
  trueDaughterEndPosZ.clear();

  trueDaughterMomentumX.clear();
  trueDaughterMomentumY.clear();
  trueDaughterMomentumZ.clear();

  trueDaughterEnergy.clear();
  trueDaughterMass.clear();

  sharedHits.clear();
  sharedHits_collection.clear();
  mcParticleHits.clear();
  mcParticleHits_collection.clear();
  hitsInRecoCluster.clear();
  hitsInRecoCluster_collection.clear();
  mcParticleEnergyByHits.clear();
  mcParticleEnergyByHits_shared_energy.clear();

  true_hit_channel.clear();
  true_hit_peakTime.clear();
  true_hit_integral.clear();

  true_hit_energy.clear();
  true_hit_spacePointX.clear();
  true_hit_spacePointY.clear();
  true_hit_spacePointZ.clear();

  //---------------------------MC Truth---------------------------//
  G4ParticleNum.clear();
  G4ParticleMother.clear();
  G4ParticleMotherPdg.clear();
  G4ParticlePdg.clear();
  
  G4ParticleEnergy.clear();
  G4ParticleMass.clear();

  G4ParticleStartPosX.clear();
  G4ParticleStartPosY.clear();
  G4ParticleStartPosZ.clear();
  
  G4ParticleEndPosX.clear();
  G4ParticleEndPosY.clear();
  G4ParticleEndPosZ.clear();

  G4ParticleMomX.clear();
  G4ParticleMomY.clear();
  G4ParticleMomZ.clear();
}


void protoana::pi0TestSelection::FillG4NTuple(const sim::ParticleList &particle_list, const simb::MCParticle* &particle, const int &number)
{
  if(fDebug)
  {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "number: " << number << std::endl;
    std::cout << "PDG code: " << particle->PdgCode() << std::endl;
    std::cout << "Energy: " << particle->E() << std::endl;
  }
  G4ParticlePdg.push_back(particle->PdgCode());
  G4ParticleEnergy.push_back(particle->E() * 1000);
  G4ParticleMass.push_back(particle->Mass() * 1000);

  TLorentzVector StartPos = particle->Position(0);
  G4ParticleStartPosX.push_back(StartPos.X());
  G4ParticleStartPosY.push_back(StartPos.Y());
  G4ParticleStartPosZ.push_back(StartPos.Z());

  TLorentzVector EndPos = particle->EndPosition();
  G4ParticleEndPosX.push_back(EndPos.X());
  G4ParticleEndPosY.push_back(EndPos.Y());
  G4ParticleEndPosZ.push_back(EndPos.Z());

  TLorentzVector momentum = particle->Momentum();
  G4ParticleMomX.push_back(momentum.X() * 1000);
  G4ParticleMomY.push_back(momentum.Y() * 1000);
  G4ParticleMomZ.push_back(momentum.Z() * 1000);

  G4ParticleNum.push_back(number);
  G4ParticleMother.push_back(particle->Mother());
  
  if(particle->Mother() != 0) // particle has mother
  {
    const simb::MCParticle* mother = particle_list.find(particle->Mother())->second;
    G4ParticleMotherPdg.push_back(mother->PdgCode());
  }
  else // particle has no mother
  {
    G4ParticleMotherPdg.push_back(-1);
  }
}


void protoana::pi0TestSelection::CollectG4Particle(const int &pdg=0, const int start=-1, const int stop=-1)
{
    const sim::ParticleList & plist = pi_serv->ParticleList();

    for(auto part = plist.begin(); part != plist.end(); part ++)
    {
      // don't compute anything till the start point
      if(start > -1 && part->first < start)
      {
        continue;
      }
      // skip the particle if it has already been added
      if(std::find(G4ParticleNum.begin(), G4ParticleNum.end(), part->first) != G4ParticleNum.end())
      {
        continue;
      }
      // finish once we process the last particle
      if(stop > -1 && part->first > stop)
      {
        if(fDebug) std::cout << "finished at: " << part->first << std::endl;
        break;
      }
      const simb::MCParticle* pPart = part->second;

      // run if a specific particle is needed
      if(pdg == pPart->PdgCode())
      {
        FillG4NTuple(plist, pPart, part->first);

        if(fDebug) std::cout << "number of Daughters: " << pPart->NumberDaughters() << std::endl;
        for (int i = pPart->FirstDaughter(); i < pPart->FirstDaughter() + pPart->NumberDaughters(); i++)
        {
          // skip the particle if it has already been added
          if(std::find(G4ParticleNum.begin(), G4ParticleNum.end(), plist.find(i)->first) != G4ParticleNum.end())
          {
            continue;
          }
          const simb::MCParticle* daughter = plist.find(i)->second;
          FillG4NTuple(plist, daughter, i);
        }
      }
      // run if all particles are needed
      if(pdg == 0)
      {
        FillG4NTuple(plist, pPart, part->first);
      }
    }
    if(fDebug) std::cout << "number of G4 particles: " << plist.size() << std::endl;
}


void protoana::pi0TestSelection::AnalyseDaughterPFP(const recob::PFParticle &daughterPFP, const art::Event &evt, const detinfo::DetectorPropertiesData &detProp, anab::MVAReader<recob::Hit,4> &hitResults)
{
  std::chrono::time_point start = std::chrono::high_resolution_clock::now();
  // get what pandora thinks the pdg code is
  pdgCodes.push_back(daughterPFP.PdgCode());

  // determine if they are track like or shower like using pandora
  // then fill a vector containing this data: 11 = shower 13 = track
  pandoraTags.push_back(PandoraIdentification(daughterPFP, evt));

  // number of hits on all planes
  unsigned int num = pfpUtil.GetNumberPFParticleHits(daughterPFP, evt, fPFParticleTag);
  nHits.push_back(num);
  if(fDebug) std::cout << "number of hits: " << num << std::endl;

  // number of collection plane hits
  const std::vector<art::Ptr<recob::Hit>> collection_hits = pfpUtil.GetPFParticleHitsFromPlane_Ptrs( daughterPFP, evt, fPFParticleTag, 2 ); // get collection plane hit objects for the daughter
  unsigned int num_collection = collection_hits.size();
  nHits_collection.push_back(num_collection);
  if(fDebug) std::cout << "number of collection plane hits: " << num_collection << std::endl;

  // slice ID
  int slice = pfpUtil.GetPFParticleSliceIndex(daughterPFP, evt, fPFParticleTag);
  sliceID.push_back(slice);
  if(fDebug) std::cout << "sliceID: " << slice << std::endl;

  // beam tag score
  int bcScore = pfpUtil.GetBeamCosmicScore(daughterPFP, evt, fPFParticleTag);
  beamCosmicScore.push_back(bcScore);
  if(fDebug) std::cout << "beam/cosmic score: " << bcScore << std::endl;

  // calculate cnn score, use collection plane hits only here as this has the better performance
  std::vector<double> cnnOutput = CNNScoreCalculator(hitResults, collection_hits, num_collection);
  CNNScore.push_back(cnnOutput[0]);
  // also output the average and em track score to calculate it in python
  emScore.push_back(cnnOutput[1]);
  trackScore.push_back(cnnOutput[2]);


  const recob::Shower* shower = 0x0; // intilise the forced shower object
  if(fDebug) std::cout << "Getting shower" << std::endl;
  // try assigning the forced shower object
  try
  {
    shower =	pfpUtil.GetPFParticleShower(daughterPFP, evt, fPFParticleTag, "pandora2Shower");

    if(shower)
    {
      if(fDebug) std::cout << "got shower" << std::endl;
      const std::vector<art::Ptr<recob::Hit> > showerHits = showerUtil.GetRecoShowerArtHits(*shower, evt, "pandora2Shower");
      art::FindManyP<recob::SpacePoint> spFromHits(showerHits, evt, fHitTag); // get space point objects of the hits

      if(fDebug) std::cout << "getting start and direction" << std::endl;
      TVector3 showerStart = shower->ShowerStart();
      TVector3 showerDir = shower->Direction();
      if(fDebug) std::cout << "got start and direction" << std::endl;

      startPosX.push_back(showerStart.X());
      startPosY.push_back(showerStart.Y());
      startPosZ.push_back(showerStart.Z());

      dirX.push_back(showerDir.X());
      dirY.push_back(showerDir.Y());
      dirZ.push_back(showerDir.Z());

      length.push_back(shower->Length());
      coneAngle.push_back(shower->OpenAngle());

      if(fRetrieveHitInfo)
      {
        std::vector<double> spx;
        std::vector<double> spy;
        std::vector<double> spz;

        std::vector<int> channel;
        std::vector<double> peakTime;
        std::vector<double> integral;

        std::vector<double> e;

        double y_assum = YPositionAssumption(showerHits, detProp, spFromHits); // y value to use if the space point is empty
        
        for(unsigned int n = 0; n < showerHits.size(); n++)
        {
          art::Ptr<recob::Hit> hit = showerHits[n];
          // hit information
          channel.push_back(hit->Channel());
          peakTime.push_back(hit->PeakTime());
          integral.push_back(hit->Integral());

          // space points
          std::vector<art::Ptr<recob::SpacePoint>> sps = spFromHits.at(n); // get nth space point
          if(!sps.empty())
          {
            TVector3 hitPoint(sps[0]->XYZ()[0], sps[0]->XYZ()[1], sps[0]->XYZ()[2]); // create space point position vector
            spx.push_back(hitPoint.X());
            spy.push_back(hitPoint.Y());
            spz.push_back(hitPoint.Z());
          }
          else
          {
            spx.push_back(-999);
            spy.push_back(-999);
            spz.push_back(-999);
          }

          // hit energy
          e.push_back(HitEnergyCalculator(hit, detProp, y_assum, sps));
        }
        hit_peakTime.push_back(peakTime);
        hit_integral.push_back(integral);
        hit_channel.push_back(channel);

        hit_energy.push_back(e);
        spacePointX.push_back(spx);
        spacePointY.push_back(spy);
        spacePointZ.push_back(spz);
      }
      else
      {
        //* Don't populate hit information if we weren't told to (even null entries)
      }

      //std::cout << "Getting calibrated shower energy" << std::endl;
      auto calo = showerUtil.GetRecoShowerCalorimetry(
          *shower, evt, "pandora2Shower", fShowerCalorimetryTag);
      bool found_calo = false;
      size_t index = 0;
      for (index = 0; index < calo.size(); ++index) {
        if (calo[index].PlaneID().Plane == 2) {
          found_calo = true;
          break;
        }
      }

      if (!found_calo) {
        calibrated_energy.push_back(-999.);
      }
      else {
        calibrated_energy.push_back(calo[index].KineticEnergy());
      }
      
      energy.push_back(ShowerEnergyCalculator(showerHits, detProp, spFromHits));
    }
    else
    {
      if(fDebug) std::cout << "couldn't get shower object! Moving on" << std::endl;
      NullRecoDaughterPFPInfo();
    }
  }
  catch( const cet::exception &e )
  {
    if(fDebug) std::cout << "couldn't get shower object! Moving on" << std::endl;
    NullRecoDaughterPFPInfo();
  }
  if(fDebug)
  {
    std::chrono::time_point stop = std::chrono::high_resolution_clock::now();
    std::cout << "analyzing PFP took: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms." << std::endl;
  }
}


void protoana::pi0TestSelection::AnalyseBeamPFP(const recob::PFParticle &beam, const art::Event &evt)
{
  if(fDebug) std::cout << "Pandora ID of beam particle: " << PandoraIdentification(beam, evt) << std::endl;
  beamSliceID = pfpUtil.GetPFParticleSliceIndex(beam, evt, fPFParticleTag);
  
  const recob::Track* beamTrack = 0x0; // set to null
  beamTrack = pfpUtil.GetPFParticleTrack(beam, evt, fPFParticleTag, fTrackerTag); // get the beam track if it exists
  
  // store beam track info
  if(!beamTrack)
  {
    if(fDebug) std::cout<< "no beam track found, moving on" << std::endl;
    NullRecoBeamInfo();
  }
  else
  {
    auto allHits = evt.getValidHandle<std::vector<recob::Hit> >(fHitTag); // perhaps move this outside method
    auto calo = trackUtil.GetRecoTrackCalorimetry(*beamTrack, evt, fTrackerTag, fBeamCalorimetryTag);
    bool found_calo = false;
    size_t i = 0;
    for (i = 0; i < calo.size(); ++i)
    {
      if (calo[i].PlaneID().Plane == 2)
      {
        found_calo = true;
        break; 
      }
    }

    if(found_calo)
    {
      auto calo_dQdX = calo[i].dQdx();
      auto TpIndices = calo[i].TpIndices();
      auto theXYZPoints = calo[i].XYZ();

      for(size_t j = 0; j < calo_dQdX.size(); ++j)
      {
        const recob::Hit & hit = (*allHits)[TpIndices[j]];
        if (hit.WireID().TPC == 5) { // not sure why special treatment is required here...
          reco_beam_calo_wire.push_back(hit.WireID().Wire + 479); // can I just get the length of the vector rather than writing the whole object to file?
        }
        else {
          reco_beam_calo_wire.push_back(hit.WireID().Wire);
        }
      }

      if(theXYZPoints.size())
      {
        //Getting the SCE corrected start/end positions & directions
        std::sort(theXYZPoints.begin(), theXYZPoints.end(), [](auto a, auto b)
            {return (a.Z() < b.Z());});

        beamStartPosX = theXYZPoints[0].X();
        beamStartPosY = theXYZPoints[0].Y();
        beamStartPosZ = theXYZPoints[0].Z();
        beamEndPosX = theXYZPoints.back().X();
        beamEndPosY = theXYZPoints.back().Y();
        beamEndPosZ = theXYZPoints.back().Z();
      }
      else
      {
        NullRecoBeamInfo();
      }

      std::vector<float> new_dEdX = calibration_SCE.GetCalibratedCalorimetry(*beamTrack, evt, fTrackerTag, fBeamCalorimetryTag, 2, -10.);
      for( size_t j = 0; j < new_dEdX.size(); ++j )
      {
        reco_beam_calibrated_dEdX_SCE.push_back( new_dEdX[j] );
      }

    }
    else
    {
      beamStartPosX = beamTrack->Trajectory().Start().X();
      beamStartPosY = beamTrack->Trajectory().Start().Y();
      beamStartPosZ = beamTrack->Trajectory().Start().Z();
      beamEndPosX = beamTrack->Trajectory().End().X();
      beamEndPosY = beamTrack->Trajectory().End().Y();
      beamEndPosZ = beamTrack->Trajectory().End().Z();
      reco_beam_calo_wire = {-999}; // Found no Calorimetry object

      // if the beam enters from the opposite direction
      if(beamStartPosZ > beamEndPosZ)
      {
        beamStartPosX = beamTrack->Trajectory().End().X();
        beamStartPosY = beamTrack->Trajectory().End().Y();
        beamStartPosZ = beamTrack->Trajectory().End().Z();
        beamEndPosX = beamTrack->Trajectory().Start().X();
        beamEndPosY = beamTrack->Trajectory().Start().Y();
        beamEndPosZ = beamTrack->Trajectory().Start().Z();
      }
    }
    std::pair<double, int> vertex_michel_score = trackUtil.GetVertexMichelScore(*beamTrack, evt, fTrackerTag, fHitTag);
    reco_beam_vertex_nHits = vertex_michel_score.second;
    reco_beam_vertex_michel_score = vertex_michel_score.first;
  }
}


void protoana::pi0TestSelection::AnalyseMCTruth(const recob::PFParticle &daughter, const art::Event &evt, const detinfo::DetectorPropertiesData &detProp, const detinfo::DetectorClocksData &clockData, const std::vector<art::Ptr<recob::Hit> > &hitVec)
{
  std::chrono::time_point start = std::chrono::high_resolution_clock::now();

  // match the MC particle assosiated to the daughter PFParticle by comparing the hit objects
  if(fDebug) std::cout << "checking shared hits" << std::endl;
  protoana::MCParticleSharedHits match = truthUtil.GetMCParticleByHits( clockData, daughter, evt, fPFParticleTag, fHitTag );
  if(fDebug) std::cout << "checked shared hits" << std::endl;
  
  if(match.particle != 0x0) // check null pointer
  {
    const sim::ParticleList & plist = pi_serv->ParticleList(); // get truth table
    const simb::MCParticle* mcParticle = match.particle; // get the MCParticle object from the match
    
    int number = -1;
    int mother = mcParticle->Mother();
    
    // get particle number
    for(auto part = plist.begin(); part != plist.end(); part ++)
    {
      if (mcParticle == part->second)
      {
        number = part->first;
        break;
      }
    }

    if(fDebug)
    {
      std::cout << "we have matched the MC particle!" << std::endl;
      std::cout << "Particle number: " << number << std::endl;
      std::cout << "Mother: " << mother << std::endl;
    }

    // particle numbers
    matchedNum.push_back(number);
    matchedMother.push_back(mother);
    if(mother == 0)
    {
      if (fDebug){std::cout << "Particle has no mother" << std::endl;}
      matchedMotherPDG.push_back(0);
    }
    else
    {
      int mother_pdg = plist.find(mother)->second->PdgCode();
      if (fDebug){std::cout << "Mother pdg code: " << mother_pdg << std::endl;}
      matchedMotherPDG.push_back(mother_pdg);
    }
    trueDaughterPDG.push_back(mcParticle->PdgCode());

    // kinematics
    trueDaughterEnergy.push_back(mcParticle->E() * 1000);
    trueDaughterMass.push_back(mcParticle->Mass() * 1000);
    
    TLorentzVector trueDaughterStartPos = mcParticle->Position(0);
    trueDaughterStartPosX.push_back(trueDaughterStartPos.X());
    trueDaughterStartPosY.push_back(trueDaughterStartPos.Y());
    trueDaughterStartPosZ.push_back(trueDaughterStartPos.Z());
    
    TLorentzVector trueDaughterEndPos = mcParticle->EndPosition();
    trueDaughterEndPosX.push_back(trueDaughterEndPos.X());
    trueDaughterEndPosY.push_back(trueDaughterEndPos.Y());
    trueDaughterEndPosZ.push_back(trueDaughterEndPos.Z());
    
    TLorentzVector trueDaughterMomentum = mcParticle->Momentum();
    trueDaughterMomentumX.push_back(trueDaughterMomentum.X() * 1000);
    trueDaughterMomentumY.push_back(trueDaughterMomentum.Y() * 1000);
    trueDaughterMomentumZ.push_back(trueDaughterMomentum.Z() * 1000);


    // hit based information
    //purity (sharedHits / recoHitsInCluster)
    //completeness (sharedHits / mcParticleHits)
    std::vector<MCParticleSharedHits> list = truthUtil.GetMCParticleListByHits(clockData, daughter, evt, fPFParticleTag, fHitTag);

    int cluster_hits = 0;
    int cluster_collection_hits = 0;
    int shared_hits = 0;
    int shared_collection_hits = 0;
    for(size_t i = 0; i < list.size(); i++)
    {
      std::vector<const recob::Hit*> hits = truthUtil.GetSharedHits(clockData, *list[i].particle, daughter, evt, fPFParticleTag, false);
      std::vector<const recob::Hit*> delta_ray_hits = truthUtil.GetSharedHits(clockData, *list[i].particle, daughter, evt, fPFParticleTag, true);
      int count = CountHitsInPlane(hits, 2) + CountHitsInPlane(delta_ray_hits, 2);
      if(list[i].particle == match.particle)
      {
        shared_hits = hits.size() + delta_ray_hits.size();
        shared_collection_hits = count;
      }
      cluster_hits += hits.size() + delta_ray_hits.size();
      cluster_collection_hits += count;
    }
    hitsInRecoCluster.push_back(cluster_hits);
    hitsInRecoCluster_collection.push_back(cluster_collection_hits);
    sharedHits.push_back(shared_hits);
    sharedHits_collection.push_back(shared_collection_hits);

    std::vector<art::Ptr<recob::Hit> > mc_art_hits = GetMCParticleArtHits( evt, clockData, *match.particle, hitVec);
    int mc_art_collection_hits = 0;
    for(size_t i = 0; i < mc_art_hits.size(); i++)
    {
      if(mc_art_hits[i]->View() == 2)
      {
        mc_art_collection_hits++;
      }
    }
    mcParticleHits.push_back(mc_art_hits.size());
    mcParticleHits_collection.push_back(mc_art_collection_hits);
    
    art::FindManyP<recob::SpacePoint> spFromHits(mc_art_hits, evt, fHitTag);
    mcParticleEnergyByHits.push_back(ShowerEnergyCalculator(mc_art_hits, detProp, spFromHits));

    // hit information
    // to correct the shared energy, we need to always extract hit channel, hit time and hit energy, then we can clear these entries after we correct mcParticleEnergyByHits
    std::vector<double> spx;
    std::vector<double> spy;
    std::vector<double> spz;

    std::vector<int> channel;
    std::vector<double> peakTime;
    std::vector<double> integral;

    std::vector<double> e;

    double y_assum = YPositionAssumption(mc_art_hits, detProp, spFromHits); // y value to use if the space point is empty

    for(size_t i = 0; i < mc_art_hits.size(); i++)
    {
      art::Ptr<recob::Hit> hit = mc_art_hits[i];
      // hit information
      channel.push_back(hit->Channel());
      peakTime.push_back(hit->PeakTime());
      integral.push_back(hit->Integral());

      // space points
      std::vector<art::Ptr<recob::SpacePoint>> sps = spFromHits.at(i);
      if(!sps.empty())
      {
        TVector3 hitPoint(sps[0]->XYZ()[0], sps[0]->XYZ()[1], sps[0]->XYZ()[2]);
        spx.push_back(hitPoint.X());
        spy.push_back(hitPoint.Y());
        spz.push_back(hitPoint.Z());
      }
      else
      {
        spx.push_back(-999);
        spy.push_back(-999);
        spz.push_back(-999);
      }

      // hit energy
      e.push_back(HitEnergyCalculator(hit, detProp, y_assum, sps));
    }
    // always push these back
    true_hit_peakTime.push_back(peakTime);
    true_hit_channel.push_back(channel);
    true_hit_energy.push_back(e);

    // this should be optional
    if(fRetrieveHitInfo)
    {
      true_hit_integral.push_back(integral);

      true_hit_spacePointX.push_back(spx);
      true_hit_spacePointY.push_back(spy);
      true_hit_spacePointZ.push_back(spz);
    }
    else
    {
      //* Don't populate hit information if we weren't told to (even null entries)
    }
  }
  else
  {
    if (fDebug) std::cout << "MC particle not matched" << std::endl;
    matchedNum.push_back(-999);
    matchedMother.push_back(-999);
    matchedMotherPDG.push_back(-999);
    trueDaughterStartPosX.push_back(-999);
    trueDaughterStartPosY.push_back(-999);
    trueDaughterStartPosZ.push_back(-999);
    trueDaughterEndPosX.push_back(-999);
    trueDaughterEndPosY.push_back(-999);
    trueDaughterEndPosZ.push_back(-999);
    trueDaughterMomentumX.push_back(-999);
    trueDaughterMomentumY.push_back(-999);
    trueDaughterMomentumZ.push_back(-999);
    trueDaughterEnergy.push_back(-999);
    trueDaughterPDG.push_back(-999);
    trueDaughterMass.push_back(-999);
    hitsInRecoCluster.push_back(-999);
    mcParticleHits.push_back(-999);
    sharedHits.push_back(-999);
    hitsInRecoCluster_collection.push_back(-999);
    mcParticleHits_collection.push_back(-999);
    sharedHits_collection.push_back(-999);
    mcParticleEnergyByHits.push_back(-999);

    true_hit_peakTime.push_back(null_double_array);
    true_hit_channel.push_back(null_int_array);
    true_hit_energy.push_back(null_double_array);

    if (fRetrieveHitInfo)
    {
      true_hit_integral.push_back(null_double_array);

      true_hit_spacePointX.push_back(null_double_array);
      true_hit_spacePointY.push_back(null_double_array);
      true_hit_spacePointZ.push_back(null_double_array);
    }
  }
  if(fDebug)
  {
    std::chrono::time_point stop = std::chrono::high_resolution_clock::now();
    std::cout << "analyzing backtracked PFP took: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms." << std::endl;
  }
}


void protoana::pi0TestSelection::AnalyseMCTruthBeam(const art::Event &evt)
{
  const simb::MCParticle* true_beam_particle = 0x0;
  auto mcTruths = evt.getValidHandle<std::vector<simb::MCTruth>>(fGeneratorTag);
  true_beam_particle = truthUtil.GetGeantGoodParticle((*mcTruths)[0],evt);
  if( !true_beam_particle ){
    if(fDebug) std::cout << "No true beam particle" << std::endl;
    return;
  }
  else
  {
    if(fDebug) std::cout << "Found true Beam particle" << std::endl;
  }

  trueBeamPdg = true_beam_particle->PdgCode();
  trueBeamMass = true_beam_particle->Mass() * 1000;
  trueBeamEnergy = true_beam_particle->E() * 1000;

  TLorentzVector trueBeamStartPos = true_beam_particle->Position(0);
  
  trueBeamStartPosX = trueBeamStartPos.X();
  trueBeamStartPosX = trueBeamStartPos.Y();
  trueBeamStartPosX = trueBeamStartPos.Z();

  TLorentzVector trueBeamEndPos = true_beam_particle->EndPosition();
  
  trueBeamEndPosX = trueBeamEndPos.X();
  trueBeamEndPosX = trueBeamEndPos.Y();
  trueBeamEndPosX = trueBeamEndPos.Z();
}

void protoana::pi0TestSelection::NullRecoBeamInfo()
{
  beamStartPosX = -999;
  beamStartPosY = -999;
  beamStartPosZ = -999;
  beamEndPosX = -999;
  beamEndPosY = -999;
  beamEndPosZ = -999;
  reco_beam_calo_wire = null_double_array;
  reco_beam_vertex_michel_score = -999;
  reco_beam_vertex_nHits = -999;
  reco_beam_calibrated_dEdX_SCE = null_double_array;
}
void protoana::pi0TestSelection::NullRecoDaughterPFPInfo()
{
  startPosX.push_back(-999);
  startPosY.push_back(-999);
  startPosZ.push_back(-999);
  dirX.push_back(-999);
  dirY.push_back(-999);
  dirZ.push_back(-999);
  length.push_back(-999);
  coneAngle.push_back(-999);
  energy.push_back(-999);
  calibrated_energy.push_back(-999.);

  // only populate null hit information if needed
  if(fRetrieveHitInfo)
  {
    spacePointX.push_back(null_double_array);
    spacePointY.push_back(null_double_array);
    spacePointZ.push_back(null_double_array);
    hit_energy.push_back(null_double_array);
    hit_peakTime.push_back(null_double_array);
    hit_integral.push_back(null_double_array);
    hit_channel.push_back(null_int_array);
  }
}

void protoana::pi0TestSelection::CorrectBacktrackedHitEnergy()
{
  std::chrono::time_point start = std::chrono::high_resolution_clock::now();

  // loop through all PFOs
  for(size_t i = 0; i < true_hit_peakTime.size(); i++)
  {
    double shared_energy = 0;
    // loop through all hits of ith PFO
    for(size_t j = 0; j < true_hit_peakTime[i].size(); j++)
    {
      if (true_hit_energy[i][j] == -999 || true_hit_peakTime[i][j] == -999 || true_hit_channel[i][j] == -999) {continue;} // escape if we are missing relevant information 
      std::map<int, int> shared_mc_particle_map; // keep record of kth PFOs that share this hit with the ith PFO

      // loop through all PFOs
      for(size_t k = 0; k < true_hit_peakTime.size(); k++)
      {
        // loop though all hits of kth PFO
        if(matchedNum[i] == matchedNum[k]) {continue;} // avoid comparing the same MCParticle
        if(shared_mc_particle_map.find(matchedNum[k]) != shared_mc_particle_map.end()) {continue;} // skip if we already checked this MCParticle
        for(size_t l = 0; l < true_hit_peakTime[k].size(); l++)
        {
          // compare hit peak time and hit channel to match PFOs
          if(true_hit_peakTime[i][j] == true_hit_peakTime[k][l] && true_hit_channel[i][j] == true_hit_channel[k][l])
          {
            // if there is a match, keep the index of the backtrackedPFO and its number in the truth table, this way we avoid double counting the same MCParticle
            shared_mc_particle_map[matchedNum[k]] = k;
            // we also found the matched hit for hit j, so we can bail out at this point
            break;
          }
        }
      }

      double numerator = trueDaughterEnergy[i];
      double denominator = numerator;

      std::map<int, int>::iterator it = shared_mc_particle_map.begin();
      while(it != shared_mc_particle_map.end())
      {
        denominator += trueDaughterEnergy[it->second];
        ++it;
      }
      double weight = 1 - (numerator / denominator);
      double shared_energy_hit = true_hit_energy[i][j] * weight;
      shared_energy += shared_energy_hit;
    }
    mcParticleEnergyByHits_shared_energy.push_back(shared_energy);
    if(fDebug) {std::cout << "backtracked MCParticle: " << i << " | true energy " << trueDaughterEnergy[i] << " | true hit energy " << mcParticleEnergyByHits[i] << " | shared hit energy :" << shared_energy << std::endl;}
  }

  if(fDebug)
  {
    std::chrono::time_point stop = std::chrono::high_resolution_clock::now();
    std::cout << "calculating shared true hit energy took: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms." << std::endl;
  }
}

void protoana::pi0TestSelection::beginJob()
{
  // intiialize output root file
  art::ServiceHandle<art::TFileService> tfs;
  fOutTree = tfs->make<TTree>("beamana", "");
  // Once we are done, write results into the ROOT file
  
  //---------------------------PFOs---------------------------//
  fOutTree->Branch("reco_PFP_ID", &PFPNum);
  fOutTree->Branch("reco_PFP_Mother", &PFPMother);
  fOutTree->Branch("reco_daughter_allShower_sliceID", &sliceID);
  fOutTree->Branch("reco_daughter_allShower_beamCosmicScore", &beamCosmicScore);
  fOutTree->Branch("reco_daughter_allSHower_PandoraPDG", &pdgCodes);

  fOutTree->Branch("pandoraTag", &pandoraTags);
  fOutTree->Branch("reco_daughter_PFP_emScore_collection", &emScore);
  fOutTree->Branch("reco_daughter_PFP_trackScore_collection", &trackScore);
  fOutTree->Branch("CNNScore_collection", &CNNScore);

  fOutTree->Branch("reco_daughter_allShower_startX", &startPosX);
  fOutTree->Branch("reco_daughter_allShower_startY", &startPosY);
  fOutTree->Branch("reco_daughter_allShower_startZ", &startPosZ);
  
  fOutTree->Branch("reco_daughter_allShower_dirX", &dirX);
  fOutTree->Branch("reco_daughter_allShower_dirY", &dirY);
  fOutTree->Branch("reco_daughter_allShower_dirZ", &dirZ);
  
  fOutTree->Branch("reco_daughter_allShower_coneAngle", &coneAngle);
  fOutTree->Branch("reco_daughter_allShower_length", &length);

  fOutTree->Branch("reco_daughter_PFP_nHits", &nHits);
  fOutTree->Branch("reco_daughter_PFP_nHits_collection", &nHits_collection);
  fOutTree->Branch("reco_daughter_allShower_energy", &energy);
  fOutTree->Branch("reco_daughter_allShower_calibrated_energy", &calibrated_energy);

  fOutTree->Branch("reco_daughter_allShower_hit_channel", &hit_channel);
  fOutTree->Branch("reco_daughter_allShower_hit_peakTime", &hit_peakTime);
  fOutTree->Branch("reco_daughter_allSHower_hit_integral", &hit_integral);

  fOutTree->Branch("reco_daughter_allShower_hit_energy", &hit_energy);
  fOutTree->Branch("reco_daughter_allShower_spacePointX", &spacePointX);
  fOutTree->Branch("reco_daughter_allShower_spacePointY", &spacePointY);
  fOutTree->Branch("reco_daughter_allShower_spacePointZ", &spacePointZ);

  //---------------------------Reconstructed Beam---------------------------//
  fOutTree->Branch("beamNum", &beam);
  fOutTree->Branch("reco_beam_sliceID", &beamSliceID);

  fOutTree->Branch("reco_beam_startX", &beamStartPosX);
  fOutTree->Branch("reco_beam_startY", &beamStartPosY);
  fOutTree->Branch("reco_beam_startZ", &beamStartPosZ);

  fOutTree->Branch("reco_beam_endX", &beamEndPosX);
  fOutTree->Branch("reco_beam_endY", &beamEndPosY);
  fOutTree->Branch("reco_beam_endZ", &beamEndPosZ);
  
  fOutTree->Branch("reco_beam_calo_wire", &reco_beam_calo_wire);
  
  fOutTree->Branch("reco_beam_vertex_nHits", &reco_beam_vertex_nHits);
  fOutTree->Branch("reco_beam_vertex_michel_score", &reco_beam_vertex_michel_score);
  
  fOutTree->Branch("reco_beam_calibrated_dEdX_SCE", &reco_beam_calibrated_dEdX_SCE);

  //---------------------------Backtracked Beam---------------------------//
  fOutTree->Branch("reco_beam_PFP_true_byHits_pdg", &trueBeamPdg);

  fOutTree->Branch("reco_beam_PFP_true_byHits_startE", &trueBeamEnergy);
  fOutTree->Branch("reco_beam_PFP_true_byHits_mass", &trueBeamMass);

  fOutTree->Branch("reco_beam_PFP_true_byHits_startX", &trueBeamStartPosX);
  fOutTree->Branch("reco_beam_PFP_true_byHits_startY", &trueBeamStartPosY);
  fOutTree->Branch("reco_beam_PFP_true_byHits_startZ", &trueBeamStartPosZ);

  fOutTree->Branch("reco_beam_PFP_true_byHits_endX", &trueBeamEndPosX);
  fOutTree->Branch("reco_beam_PFP_true_byHits_endY", &trueBeamEndPosY);
  fOutTree->Branch("reco_beam_PFP_true_byHits_endZ", &trueBeamEndPosZ);


  //---------------------------Backtracked Daughter---------------------------//
  fOutTree->Branch("reco_daughter_PFP_true_byHits_ID", &matchedNum);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_Mother", &matchedMother);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_pdg", &trueDaughterPDG);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_Mother_pdg", &matchedMotherPDG);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_startX", &trueDaughterStartPosX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_startY", &trueDaughterStartPosY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_startZ", &trueDaughterStartPosZ);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_endX", &trueDaughterEndPosX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_endY", &trueDaughterEndPosY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_endZ", &trueDaughterEndPosZ);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_pX", &trueDaughterMomentumX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_pY", &trueDaughterMomentumY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_pZ", &trueDaughterMomentumZ);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_startE", &trueDaughterEnergy);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_mass", &trueDaughterMass);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_sharedHits", &sharedHits);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_sharedHits_collection", &sharedHits_collection);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_nHits", &mcParticleHits);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_nHits_collection", &mcParticleHits_collection);  
  fOutTree->Branch("reco_daughter_PFP_true_byHits_hitsInRecoCluster", &hitsInRecoCluster);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_hitsInRecoCluster_collection", &hitsInRecoCluster_collection);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_EnergyByHits", &mcParticleEnergyByHits);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_EnergyByHits_correction", &mcParticleEnergyByHits_shared_energy);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_hit_channel", & true_hit_channel);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_hit_peakTime", & true_hit_peakTime);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_hit_integral", & true_hit_integral);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_hit_energy", & true_hit_energy);  
  fOutTree->Branch("reco_daughter_PFP_true_byHits_spacePointX", &true_hit_spacePointX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_spacePointY", &true_hit_spacePointY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_spacePointZ", &true_hit_spacePointZ);

  //---------------------------MC Truth---------------------------//
  fOutTree->Branch("g4_num", &G4ParticleNum);
  fOutTree->Branch("g4_mother", &G4ParticleMother);
  fOutTree->Branch("g4_Pdg", &G4ParticlePdg);
  fOutTree->Branch("g4_mother_Pdg", &G4ParticleMotherPdg);

  fOutTree->Branch("g4_startE", &G4ParticleEnergy);
  fOutTree->Branch("g4_mass", &G4ParticleMass);

  fOutTree->Branch("g4_startX", &G4ParticleStartPosX);
  fOutTree->Branch("g4_startY", &G4ParticleStartPosY);
  fOutTree->Branch("g4_startZ", &G4ParticleStartPosZ);

  fOutTree->Branch("g4_endX", &G4ParticleEndPosX);
  fOutTree->Branch("g4_endY", &G4ParticleEndPosY);
  fOutTree->Branch("g4_endZ", &G4ParticleEndPosZ);

  fOutTree->Branch("g4_pX", &G4ParticleMomX);
  fOutTree->Branch("g4_pY", &G4ParticleMomY);
  fOutTree->Branch("g4_pZ", &G4ParticleMomZ);


  //---------------------------Event info---------------------------//
  fOutTree->Branch("Run", &run);
  fOutTree->Branch("SubRun", &subRun);
  fOutTree->Branch("EventID", &eventID);

  fOutTree->Branch("totalEvents", &totalEvents);
  fOutTree->Branch("beamEvents", &beamEvents);

}


void protoana::pi0TestSelection::analyze(art::Event const & evt)
{
  std::chrono::time_point analyze_start = std::chrono::high_resolution_clock::now();
  //-----------------------------------------------//
  std::cout << "module running..." << std::endl;
  reset(); // clear any outputs that are lists
  
  // print metadata
  run = evt.run();
  subRun = evt.subRun();
  eventID = evt.id().event();
  std::cout << "run: " << run << std::endl;
  std::cout << "subrun: " << subRun << std::endl;
  std::cout << "event: " << eventID << std::endl;
  //-----------------------------------------------//

  //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
  // get various information needed to retrieve necessary data
  if(fDebug) std::cout << "getting PFP handle" << std::endl;
  art::ValidHandle<std::vector<recob::PFParticle> > pfpVec = evt.getValidHandle<std::vector<recob::PFParticle> >( fPFParticleTag ); // object to allow us to reference the PFParticles in the event
  std::cout << "number of PFParticles: " << pfpVec->size() << std::endl;
  if(fDebug) std::cout << "got PFP handle" << std::endl;

  if(fDebug) std::cout << "getting hit handle" << std::endl;
  art::Handle<std::vector<recob::Hit> > hitHandle = evt.getHandle<std::vector<recob::Hit> >(fHitTag); // handle takes no PFParticle in reference, so is the same for each MCParticle, move outside loop into main function
  if (!hitHandle)
  {
    std::cout << "could not find hits in event." << std::endl;
  }
  std::vector<art::Ptr<recob::Hit> > hitVec;
  art::fill_ptr_vector(hitVec, hitHandle); // this might be fairly taxing to do for every mcParticle we look at, so perhaps do it one time and pass to method
  if(fDebug) std::cout << "got hit handle" << std::endl;


  if(fDebug) std::cout << "getting clockData" << std::endl;
  auto const clockData = art::ServiceHandle<detinfo::DetectorClocksService>()->DataFor(evt); // use timing to match PFP to MC
  if(fDebug) std::cout << "got clockData" << std::endl;
  auto const detProp =  art::ServiceHandle<detinfo::DetectorPropertiesService>()->DataFor(evt, clockData); // object containing physical proteties of the detector

  anab::MVAReader<recob::Hit,4> hitResults(evt, "emtrkmichelid:emtrkmichel"); // info for CNN score calculation
  //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//

  //-------------------------------------------------------------------------------------------//
  // checks beam event info from the generator and data, not used for particle gun tests
  if(!fPi0Only)
  {
    std::vector<art::Ptr<beam::ProtoDUNEBeamEvent>> beamVec;
    try
    {
      auto beamHandle = evt.getValidHandle<std::vector<beam::ProtoDUNEBeamEvent>>("generator");
      if( beamHandle.isValid())
      {
        art::fill_ptr_vector(beamVec, beamHandle);
      }
    }
    catch (const cet::exception &e)
    {
      std::cout << "BeamEvent generator object not found, moving on" << std::endl;
      return;
    }
  }
  //-------------------------------------------------------------------------------------------//

  //-------------------------------------------------------------------------------------------------------------//
  // Get an array of beam particles
  std::vector<const recob::PFParticle*> beamParticles = pfpUtil.GetPFParticlesFromBeamSlice(evt, fPFParticleTag);
  
  beam = -999;
  if(beamParticles.size() == 0)
  {
    std::cout << "no beam particle..." << std::endl;
  }
  else
  {
    beam = beamParticles[0]->Self();
  }
  if(beamParticles.size() > 1)
  {
    std::cout << "there shouldn't be more than one beam particle" << std::endl;
  }
  //-------------------------------------------------------------------------------------------------------------//

  //-------------------------------------------------------------------------------------------------------------//
  // main code

  // get track-like beam info, not for pi0 only MC events
  if(!fPi0Only && beamParticles.size() != 0)
  {
    AnalyseBeamPFP(*beamParticles[0], evt);
    if(!evt.isRealData())
    {
      AnalyseMCTruthBeam(evt);
    }
    else
    {
      trueBeamPdg = -999;
      trueBeamMass = -999;
      trueBeamEnergy = -999;
      trueBeamStartPosX = -999; 
      trueBeamStartPosX = -999;
      trueBeamStartPosX = -999;
      trueBeamEndPosX = -999;
      trueBeamEndPosX = -999;
      trueBeamEndPosX = -999;
    }
  }
  else
  {
    NullRecoBeamInfo();
  }

  for(recob::PFParticle pfp : *pfpVec)
  {
    const int self = pfp.Self();
    int parent = pfp.Parent();

    // make it clear that the particle has no parent
    if(pfp.Parent() > pfpVec->size())
    {
      parent = -999;
    }

    // print some information for debugging
    if(fDebug)
    {
      std::cout << "----------------------------------------" << std::endl;
      std::cout << "PFP number: " << self << std::endl;
      std::cout << "is primary? " << pfp.IsPrimary() << std::endl;
      std::cout << "Number of daughters: " << pfp.NumDaughters() << std::endl;
      if(self == beam)
      {
        std::cout << "beam particle: " << self << std::endl;
      }
      std::cout << "parent: " << parent << std::endl;
    }
    

    PFPNum.push_back(self);
    PFPMother.push_back(parent);

    AnalyseDaughterPFP(pfp, evt, detProp, hitResults);
    if(!evt.isRealData())
    {
      AnalyseMCTruth(pfp, evt, detProp, clockData, hitVec);
    }
    if(fDebug) std::cout << "----------------------------------------" << std::endl;
  }
  
  // Collect information from truth tables depending on which reco files are analysed.
  if(!evt.isRealData())
  {
    switch (fRetrieveG4)
    {
    case ALL:
      std::cout << "Retreiving ALL MCParticles" << std::endl;
      CollectG4Particle();
      break;
    
    case PI0:
      std::cout << "Retreiving all pi0 MCParticles + daughters" << std::endl;
      CollectG4Particle(111);
      break;
    case PI0_PIP:
      std::cout << "Retreiving all pi0 MCParticles + daughters" << std::endl;
      CollectG4Particle(111);
      std::cout << "Retreiving all pi+ MCParticles + daughters" << std::endl;
      CollectG4Particle(211);
      break;
    case DIPHOTON:
      std::cout << "Retreiving photons 0 and 1 + daughters" << std::endl;
      CollectG4Particle(22, 0, 2);
      break;

    case NONE:
      std::cout << "Retreiving no MCParticles from particle list" << std::endl;
      break;

    default:
      std::cout << "Retreiving no MCParticles from particle list" << std::endl;
      break;
    }
  }
  //-------------------------------------------------------------------------------------------------------------//

  CorrectBacktrackedHitEnergy(); // some processing
  
  if(!fRetrieveHitInfo)// clear these vectors if we don't want to write hit information to file.
  {
    true_hit_peakTime.clear();
    true_hit_channel.clear();
    true_hit_energy.clear();
  }

  totalEvents++;
  fOutTree->Fill(); // fill the root tree with the outputs
  std::chrono::time_point analyze_stop = std::chrono::high_resolution_clock::now();
  std::cout << "analyze took: " << std::chrono::duration_cast<std::chrono::seconds>(analyze_stop - analyze_start).count() << " seconds." << std::endl;
}

// Maybe do some stuff here???
void protoana::pi0TestSelection::endJob()
{

}

DEFINE_ART_MODULE(protoana::pi0TestSelection)
