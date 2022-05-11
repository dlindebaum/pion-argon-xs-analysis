////////////////////////////////////////////////////////////////////////
// Class:       pi0TestSelection
// Plugin Type: analyzer (art v2_07_03)
// File:        pi0TestSelection_module.cc
//TODO follow new calibration as done in PDSPAnalyser (after understanding current reconsturction)
//? output raw CNN outputs?
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
#include "dune/DuneObj/ProtoDUNEBeamEvent.h"

//ROOT includes
#include <TTree.h>
#include <TH1D.h>
#include "Math/Vector3D.h"

//Use this to get CNN output
#include "lardata/ArtDataHelper/MVAReader.h"

#include <algorithm>

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
  int PandoraIdentification(const recob::PFParticle &daughterPFP, const art::Event &evt);
  std::vector<double> CNNScoreCalculator(anab::MVAReader<recob::Hit,4> &hitResults, const std::vector< art::Ptr< recob::Hit > > &hits, unsigned int &n);
  std::vector<double> StartHitQuantityCalculator(TVector3 &hitStart, TVector3 &hit, TVector3 &direction);
  std::vector<std::vector<double> > StartHits(unsigned int n_hits, art::FindManyP<recob::SpacePoint> spFromHits, TVector3 &showerStart, TVector3 &direction);
  double ShowerEnergyCalculator(const std::vector<art::Ptr<recob::Hit> > &hits, const detinfo::DetectorPropertiesData &detProp, art::FindManyP<recob::SpacePoint> &spFromHits);
  void reset();

  void AnalyseDaughterPFP(const recob::PFParticle &daughterPFP, const art::Event &evt, const detinfo::DetectorPropertiesData &detProp, anab::MVAReader<recob::Hit,4> &hitResults);
  void AnalyseBeamPFP(const recob::PFParticle &beam, const art::Event &evt);
  void AnalyseMCTruth(const recob::PFParticle &daughter, const art::Event &evt, const detinfo::DetectorClocksData &clockData);
  void AnalyseMCTruthBeam(const art::Event &evt);
  void FillG4NTuple(const simb::MCParticle* &particle, const int &number);
  void CollectG4Particle(const int &Pdg, const int start, const int stop);

  void AnalyseFromBeam(const art::Event &evt, const detinfo::DetectorClocksData &clockData, const detinfo::DetectorPropertiesData &detProp, anab::MVAReader<recob::Hit,4> &hitResults, std::vector<recob::PFParticle> pfpVec);

  private:
  
  enum G4Mode{PI0=1, DIPHOTON=2, ALL=3, NONE=0}; // determines what MC particles are retrieved from the truth table
  
  // fcl parameters, order matters!
  protoana::ProtoDUNECalibration calibration_SCE;
  std::string fCalorimetryTag;
  std::string fTrackerTag;
  std::string fShowerTag;
  std::string fHitTag;
  std::string fPFParticleTag;
  std::string fGeneratorTag;
  protoana::ProtoDUNEBeamlineUtils fBeamlineUtils; // get BeamLineUtils class... <consider removing>
  art::ServiceHandle<geo::Geometry> geom;
  art::ServiceHandle<cheat::BackTrackerService> bt_serv;
  art::ServiceHandle< cheat::ParticleInventoryService > pi_serv;

  bool fPi0Only;
  bool fHitSpacePoints;
  G4Mode fRetrieveG4;

  //Initialise protodune analysis utility classes
  protoana::ProtoDUNEPFParticleUtils pfpUtil;
  protoana::ProtoDUNETrackUtils trkUtil;
  protoana::ProtoDUNETruthUtils truthUtil;
  protoana::ProtoDUNEShowerUtils showerUtil;
  protoana::ProtoDUNETrackUtils trackUtil;

  // local variables
  TTree *fOutTree = new TTree;

  // meta-data
  double totalEvents; // number of events processed
  double beamEvents; // number of events with beam particles
  std::vector<int> pdgCodes; // particle pdg codes

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

  // shower angle
  std::vector<double> coneAngle;
  //shower length
  std::vector<double> length;

  // hit/energy quantities
  std::vector<int> nHits; // number of collection plane hits
  std::vector<double> energy; // reco shower energy in ???

  // quantity used to calculate the number of start hits
  std::vector<std::vector<double>> hitRadial;
  std::vector<std::vector<double>> hitLongitudinal;
  std::vector<std::vector<double>> spacePointX;
  std::vector<std::vector<double>> spacePointY;
  std::vector<std::vector<double>> spacePointZ;

  // beam start position
  double beamStartPosX;
  double beamStartPosY;
  double beamStartPosZ;

  // beam end position
  double beamEndPosX;
  double beamEndPosY;
  double beamEndPosZ;

  int trueBeamPdg;
  std::vector<int> trueDaughterPdg;

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

  std::vector<double> trueDaughterEnergy; // mc shower energy in GeV
  std::vector<double> trueDaughterMass;

  // true parent start positions
  std::vector<double> trueParentStartPosX;
  std::vector<double> trueParentStartPosY;
  std::vector<double> trueParentStartPosZ;

  // true parent end positions
  std::vector<double> trueParentEndPosX;
  std::vector<double> trueParentEndPosY;
  std::vector<double> trueParentEndPosZ;

  // true parent momentum
  std::vector<double> trueParentMomentumX;
  std::vector<double> trueParentMomentumY;
  std::vector<double> trueParentMomentumZ;

  std::vector<double> trueParentEnergy; // mc shower energy in GeV
  std::vector<double> trueParentMass;
  std::vector<int> trueParentPdg;

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


  std::vector<int> G4ParticleNum;
  std::vector<int> G4ParticleMother;

  std::vector<int> matchedG4DaughterPdg;
  std::vector<int> matchedG4ParentPdg;

  std::vector<int> PFPNum;
  std::vector<int> PFPMother;

  unsigned int eventID;
  unsigned int run;
  unsigned int subRun;
  int beam;
};


protoana::pi0TestSelection::pi0TestSelection(fhicl::ParameterSet const & p)
  :
  EDAnalyzer(p),
  calibration_SCE(p.get<fhicl::ParameterSet>("CalibrationParsSCE")),
  fCalorimetryTag(p.get<std::string>("CalorimetryTag")),
  fTrackerTag(p.get<std::string>("TrackerTag")),
  fShowerTag(p.get<std::string>("ShowerTag")),
  fHitTag(p.get<std::string>("HitTag")),
  fPFParticleTag(p.get<std::string>("PFParticleTag")),
  fGeneratorTag(p.get<std::string>("GeneratorTag")),
  fBeamlineUtils(p.get<fhicl::ParameterSet>("BeamlineUtils")),
  fPi0Only(p.get<bool>("Pi0Only")),
  fHitSpacePoints(p.get<bool>("RetrieveSpacePoints")),
  fRetrieveG4(static_cast<G4Mode>(p.get<int>("RetrieveG4")))
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
    std::cout << "could not reconstruct energy" << std::endl;
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


// calculates the quantities for determining hits close to the shower start
std::vector<double> protoana::pi0TestSelection::StartHitQuantityCalculator(TVector3 &hitStart, TVector3 &hit, TVector3 &direction)
{
  std::vector<double> output{};  // make the return type a pair!
  TVector3 cross = (hit - hitStart).Cross(direction); // rsin(theta) compare to cylinder radius
  double dot = (hit - hitStart).Dot(direction); // rcos(theta) compare to cylinder length
  output.push_back(cross.Mag());
  output.push_back(dot);
  return output;
}

std::vector<std::vector<double> > protoana::pi0TestSelection::StartHits(unsigned int n_hits, art::FindManyP<recob::SpacePoint> spFromHits, TVector3 &showerStart, TVector3 &direction)
{
  std::vector<std::vector<double> > out;
  // calculates quantities needed to compute the start hits
  std::vector<double> hitRad; // magnitudes of cross product of hit positions and shower direction
  std::vector<double> hitLong; // dot product of of hit positions and shower direction
  for(unsigned int n = 0; n < n_hits; n++)
  {
    std::vector<art::Ptr<recob::SpacePoint>> sps = spFromHits.at(n); // get nth space point

    if(!sps.empty())
    {
      TVector3 hitPoint(sps[0]->XYZ()[0], sps[0]->XYZ()[1], sps[0]->XYZ()[2]); // create space point position vector

      std::vector<double> startHitQuantities = StartHitQuantityCalculator(showerStart, hitPoint, direction); // get start hit quantities

      hitRad.push_back(startHitQuantities[0]);
      hitLong.push_back(startHitQuantities[1]);
    }
    else
    {
      hitRad.push_back(-999);
      hitLong.push_back(-999);
    }
  }
  out.push_back(hitRad);
  out.push_back(hitLong);
  return out;
}

// Clears the various analyser outputs at the start of a new event to remove the previous events contents
void protoana::pi0TestSelection::reset()
{
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

  energy.clear();
  nHits.clear();

  hitRadial.clear();
  hitLongitudinal.clear();
  spacePointX.clear();
  spacePointY.clear();
  spacePointZ.clear();

  trueDaughterPdg.clear();
  trueDaughterMass.clear();
  trueDaughterEnergy.clear();

  trueDaughterStartPosX.clear();
  trueDaughterStartPosY.clear();
  trueDaughterStartPosZ.clear();
  
  trueDaughterEndPosX.clear();
  trueDaughterEndPosY.clear();
  trueDaughterEndPosZ.clear();

  trueDaughterMomentumX.clear();
  trueDaughterMomentumY.clear();
  trueDaughterMomentumZ.clear();

  trueParentPdg.clear();
  trueParentMass.clear();
  trueParentEnergy.clear();

  trueParentStartPosX.clear();
  trueParentStartPosY.clear();
  trueParentStartPosZ.clear();
  
  trueParentEndPosX.clear();
  trueParentEndPosY.clear();
  trueParentEndPosZ.clear();

  trueParentMomentumX.clear();
  trueParentMomentumY.clear();
  trueParentMomentumZ.clear();

  G4ParticlePdg.clear();
  G4ParticleMass.clear();
  G4ParticleEnergy.clear();

  G4ParticleStartPosX.clear();
  G4ParticleStartPosY.clear();
  G4ParticleStartPosZ.clear();
  
  G4ParticleMomX.clear();
  G4ParticleMomY.clear();
  G4ParticleMomZ.clear();

  G4ParticleEndPosX.clear();
  G4ParticleEndPosY.clear();
  G4ParticleEndPosZ.clear();
  G4ParticleNum.clear();
  G4ParticleMother.clear();

  matchedG4DaughterPdg.clear();
  matchedG4ParentPdg.clear();

  PFPNum.clear();
  PFPMother.clear();
}

void protoana::pi0TestSelection::FillG4NTuple(const simb::MCParticle* &particle, const int &number)
{
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "number: " << number << std::endl;
  std::cout << "PDG code: " << particle->PdgCode() << std::endl;
  std::cout << "Energy: " << particle->E() << std::endl;
  G4ParticlePdg.push_back(particle->PdgCode());
  G4ParticleEnergy.push_back(particle->E());
  G4ParticleMass.push_back(particle->Mass());

  TLorentzVector StartPos = particle->Position(0);
  G4ParticleStartPosX.push_back(StartPos.X());
  G4ParticleStartPosY.push_back(StartPos.Y());
  G4ParticleStartPosZ.push_back(StartPos.Z());

  TLorentzVector EndPos = particle->EndPosition();
  G4ParticleEndPosX.push_back(EndPos.X());
  G4ParticleEndPosY.push_back(EndPos.Y());
  G4ParticleEndPosZ.push_back(EndPos.Z());

  TLorentzVector momentum = particle->Momentum();
  G4ParticleMomX.push_back(momentum.X());
  G4ParticleMomY.push_back(momentum.Y());
  G4ParticleMomZ.push_back(momentum.Z());

  G4ParticleNum.push_back(number);
  G4ParticleMother.push_back(particle->Mother());
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
      // finish once we process the last particle
      if(stop > -1 && part->first > stop)
      {
        std::cout << "finished at: " << part->first << std::endl;
        break;
      }
      const simb::MCParticle* pPart = part->second;
      // run if a specific particle is needed
      if(pdg == pPart->PdgCode())
      {
        FillG4NTuple(pPart, part->first);

        std::cout << "number of Daughters: " << pPart->NumberDaughters() << std::endl;
        for (int i = pPart->FirstDaughter(); i < pPart->FirstDaughter() + pPart->NumberDaughters(); i++)
        {
          const simb::MCParticle* daughter = plist.find(i)->second;
          FillG4NTuple(daughter, i);
        }
      }
      // run if all particles are needed
      if(pdg == 0)
      {
        FillG4NTuple(pPart, part->first);
      }
    }
    std::cout << "number of G4 particles: " << plist.size() << std::endl;
}


void protoana::pi0TestSelection::AnalyseDaughterPFP(const recob::PFParticle &daughterPFP, const art::Event &evt, const detinfo::DetectorPropertiesData &detProp, anab::MVAReader<recob::Hit,4> &hitResults)
{
  // get what pandora thinks the pdg code is
  pdgCodes.push_back(daughterPFP.PdgCode());

  // determine if they are track like or shower like using pandora
  // then fill a vector containing this data: 11 = shower 13 = track
  pandoraTags.push_back(PandoraIdentification(daughterPFP, evt));

  // number of collection plane hits
  auto collection_hits = pfpUtil.GetPFParticleHitsFromPlane_Ptrs( daughterPFP, evt, fPFParticleTag, 2 ); // get collection plane hit objects for the daughter
  unsigned int num = collection_hits.size();
  nHits.push_back(num);
  std::cout << "collection plane hits: " << num << std::endl;


  // calculate cnn score
  std::vector<double> cnnOutput = CNNScoreCalculator(hitResults, collection_hits, num);
  CNNScore.push_back(cnnOutput[0]);
  // also output the average and em track score to calculate it in python
  emScore.push_back(cnnOutput[1]);
  trackScore.push_back(cnnOutput[2]);


  const recob::Shower* shower = 0x0; // intilise the forced shower object
  std::cout << "Getting shower" << std::endl;
  // try assigning the forced shower object
  try
  {
    shower =	pfpUtil.GetPFParticleShower(daughterPFP, evt, fPFParticleTag, "pandora2Shower");

    if(shower)
    {
      std::cout << "got shower" << std::endl;
      const std::vector<art::Ptr<recob::Hit> > showerHits = showerUtil.GetRecoShowerArtHits(*shower, evt, "pandora2Shower");
      art::FindManyP<recob::SpacePoint> spFromHits(showerHits, evt, fHitTag); // get space point objects of the hits

      std::cout << "getting start and direction" << std::endl;
      TVector3 showerStart = shower->ShowerStart();
      TVector3 showerDir = shower->Direction();
      std::cout << "got start and direction" << std::endl;

      startPosX.push_back(showerStart.X());
      startPosY.push_back(showerStart.Y());
      startPosZ.push_back(showerStart.Z());

      dirX.push_back(showerDir.X());
      dirY.push_back(showerDir.Y());
      dirZ.push_back(showerDir.Z());

      length.push_back(shower->Length());
      coneAngle.push_back(shower->OpenAngle());

      // calculates quantities needed to compute the start hits <move to function?>
      std::vector<double> hitRad; // magnitudes of cross product of hit positions and shower direction
      std::vector<double> hitLong; // dot product of of hit positions and shower direction
      std::vector<double> spx;
      std::vector<double> spy;
      std::vector<double> spz;
      for(unsigned int n = 0; n < num; n++)
      {
        std::vector<art::Ptr<recob::SpacePoint>> sps = spFromHits.at(n); // get nth space point

        if(!sps.empty())
        {
          TVector3 hitPoint(sps[0]->XYZ()[0], sps[0]->XYZ()[1], sps[0]->XYZ()[2]); // create space point position vector

          std::vector<double> startHitQuantities = StartHitQuantityCalculator(showerStart, hitPoint, showerDir); // get start hit quantities

          if(fHitSpacePoints)
          {
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
          hitRad.push_back(startHitQuantities[0]);
          hitLong.push_back(startHitQuantities[1]);
        }
        else
        {
          spx.push_back(-999);
          spy.push_back(-999);
          spz.push_back(-999);
          hitRad.push_back(-999);
          hitLong.push_back(-999);
        }
      }

      hitRadial.push_back(hitRad);
      hitLongitudinal.push_back(hitLong);
      spacePointX.push_back(spx);
      spacePointY.push_back(spy);
      spacePointZ.push_back(spz);

      // calculate and push back shower energy
      energy.push_back(ShowerEnergyCalculator(showerHits, detProp, spFromHits));
    }
    else
    {
      std::cout << "couldn't get shower object! Moving on" << std::endl;
      startPosX.push_back(-999);
      startPosY.push_back(-999);
      startPosZ.push_back(-999);
      std::vector<double> null (1, -999);
      hitRadial.push_back( null );
      hitLongitudinal.push_back( null );
      spacePointX.push_back( null );
      spacePointY.push_back( null );
      spacePointZ.push_back( null );
      dirX.push_back(-999);
      dirY.push_back(-999);
      dirZ.push_back(-999);
      length.push_back(-999);
      coneAngle.push_back(-999);
      energy.push_back(-999);
    }
  }
  catch( const cet::exception &e )
  {
    std::cout << "couldn't get shower object! Moving on" << std::endl;
    startPosX.push_back(-999);
    startPosY.push_back(-999);
    startPosZ.push_back(-999);
    dirX.push_back(-999);
    dirY.push_back(-999);
    dirZ.push_back(-999);
    length.push_back(-999);
    coneAngle.push_back(-999);
    energy.push_back(-999);
  }
}


void protoana::pi0TestSelection::AnalyseBeamPFP(const recob::PFParticle &beam, const art::Event &evt)
{
  const recob::Track* beamTrack = 0x0; // set to null
  beamTrack = pfpUtil.GetPFParticleTrack(beam, evt, fPFParticleTag, fTrackerTag); // get the beam track if it exists
  
  std::cout << "Pandora ID of beam particle: " << PandoraIdentification(beam, evt) << std::endl;

  // store beam track info
  if(!beamTrack)
  {
    std::cout<< "no beam track found, moving on" << std::endl;
    beamStartPosX = -999;
    beamStartPosY = -999;
    beamStartPosZ = -999;
    beamEndPosX = -999;
    beamEndPosY = -999;
    beamEndPosZ = -999;
  }
  else
  {
    beamStartPosX = beamTrack->Trajectory().Start().X();
    beamStartPosY = beamTrack->Trajectory().Start().Y();
    beamStartPosZ = beamTrack->Trajectory().Start().Z();
    beamEndPosX = beamTrack->Trajectory().End().X();
    beamEndPosY = beamTrack->Trajectory().End().Y();
    beamEndPosZ = beamTrack->Trajectory().End().Z();

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
}


void protoana::pi0TestSelection::AnalyseMCTruth(const recob::PFParticle &daughter, const art::Event &evt, const detinfo::DetectorClocksData &clockData)
{
  std::cout << "getting shared hits" << std::endl;
  // match the MC particle assosiated to the daughter PFParticle by comparing the hit objects
  protoana::MCParticleSharedHits match = truthUtil.GetMCParticleByHits( clockData, daughter, evt, fPFParticleTag, fHitTag );
  std::cout << "got shared hits" << std::endl;
  const simb::MCParticle* mcParticle = match.particle; // get the MCParticle object from the match
  const sim::ParticleList & plist = pi_serv->ParticleList(); // get particle list, g4?
  if(mcParticle)
  {
    std::cout << "we have matched the MC particles!" << std::endl;
    const simb::MCParticle * parent = plist.Primary(mcParticle->Mother());
    std::cout << "parent PID" << std::endl;
    std::cout << parent->PdgCode() << std::endl;

    const simb::MCParticle * g4Particle = truthUtil.MatchPduneMCtoG4(*mcParticle, evt);
    const simb::MCParticle * g4ParticleParent = truthUtil.MatchPduneMCtoG4(*parent, evt);
    matchedG4DaughterPdg.push_back(g4Particle->PdgCode());
    matchedG4ParentPdg.push_back(g4ParticleParent->PdgCode());
    

    TLorentzVector trueDaughterStartPos = mcParticle->Position(0);
    trueDaughterStartPosX.push_back(trueDaughterStartPos.X());
    trueDaughterStartPosY.push_back(trueDaughterStartPos.Y());
    trueDaughterStartPosZ.push_back(trueDaughterStartPos.Z());
    TLorentzVector trueDaughterEndPos = mcParticle->EndPosition();
    trueDaughterEndPosX.push_back(trueDaughterEndPos.X());
    trueDaughterEndPosY.push_back(trueDaughterEndPos.Y());
    trueDaughterEndPosZ.push_back(trueDaughterEndPos.Z());
    TLorentzVector trueDaughterMomentum = mcParticle->Momentum();
    trueDaughterMomentumX.push_back(trueDaughterMomentum.X());
    trueDaughterMomentumY.push_back(trueDaughterMomentum.Y());
    trueDaughterMomentumZ.push_back(trueDaughterMomentum.Z());
    
    trueDaughterEnergy.push_back(mcParticle->E());
    trueDaughterPdg.push_back(mcParticle->PdgCode());
    trueDaughterMass.push_back(mcParticle->Mass());

    TLorentzVector trueParentStartPos = parent->Position(0);
    trueParentStartPosX.push_back(trueParentStartPos.X());
    trueParentStartPosY.push_back(trueParentStartPos.Y());
    trueParentStartPosZ.push_back(trueParentStartPos.Z());
    TLorentzVector trueParentEndPos = parent->EndPosition();
    trueParentEndPosX.push_back(trueParentEndPos.X());
    trueParentEndPosY.push_back(trueParentEndPos.Y());
    trueParentEndPosZ.push_back(trueParentEndPos.Z());
    TLorentzVector trueParentMomentum = parent->Momentum();
    trueParentMomentumX.push_back(trueParentMomentum.X());
    trueParentMomentumY.push_back(trueParentMomentum.Y());
    trueParentMomentumZ.push_back(trueParentMomentum.Z());
    
    trueParentEnergy.push_back(parent->E());
    trueParentPdg.push_back(parent->PdgCode());
    trueParentMass.push_back(parent->Mass());
  }
  else
  {
    std::cout << "MC particle not matched" << std::endl;
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
    trueDaughterPdg.push_back(-999);
    trueDaughterMass.push_back(-999);

    trueParentStartPosX.push_back(-999);
    trueParentStartPosY.push_back(-999);
    trueParentStartPosZ.push_back(-999);
    trueParentEndPosX.push_back(-999);
    trueParentEndPosY.push_back(-999);
    trueParentEndPosZ.push_back(-999);
    trueParentMomentumX.push_back(-999);
    trueParentMomentumY.push_back(-999);
    trueParentMomentumZ.push_back(-999);
    trueParentEnergy.push_back(-999);
    trueParentPdg.push_back(-999);
    trueParentMass.push_back(-999);

    matchedG4DaughterPdg.push_back(-999);
    matchedG4ParentPdg.push_back(-999);
  }
}


void protoana::pi0TestSelection::AnalyseMCTruthBeam(const art::Event &evt)
{
  const simb::MCParticle* true_beam_particle = 0x0;
  auto mcTruths = evt.getValidHandle<std::vector<simb::MCTruth>>(fGeneratorTag);
  true_beam_particle = truthUtil.GetGeantGoodParticle((*mcTruths)[0],evt);
  if( !true_beam_particle ){
    std::cout << "No true beam particle" << std::endl;
    return;
  }
  else
  {
    std::cout << "Found true Beam particle" << std::endl;
  }

  trueBeamPdg = true_beam_particle->PdgCode();
  trueBeamMass = true_beam_particle->Mass();
  trueBeamEnergy = true_beam_particle->E();

  TLorentzVector trueBeamStartPos = true_beam_particle->Position(0);
  
  trueBeamStartPosX = trueBeamStartPos.X();
  trueBeamStartPosX = trueBeamStartPos.Y();
  trueBeamStartPosX = trueBeamStartPos.Z();

  TLorentzVector trueBeamEndPos = true_beam_particle->EndPosition();
  
  trueBeamEndPosX = trueBeamEndPos.X();
  trueBeamEndPosX = trueBeamEndPos.Y();
  trueBeamEndPosX = trueBeamEndPos.Z();
}


void protoana::pi0TestSelection::AnalyseFromBeam(art::Event const &evt, const detinfo::DetectorClocksData &clockData, const detinfo::DetectorPropertiesData &detProp, anab::MVAReader<recob::Hit,4> &hitResults, std::vector<recob::PFParticle> pfpVec)
{
  // Get only Beam particle by checking the Beam slices
  std::vector<const recob::PFParticle*> beamParticles = pfpUtil.GetPFParticlesFromBeamSlice(evt, fPFParticleTag);
  
  if(beamParticles.size() == 0)
  {
    std::cout << "no beam particle, moving on..." << std::endl;
    totalEvents ++;
    return;
  }
  if(beamParticles.size() > 1)
  {
    std::cout << "there shouldn't be more than one beam particle" << std::endl;
  }
  auto beamParticle = beamParticles[0]; // get the first beam particle
  beamEvents ++;

  // get track-like beam info, not for pi0 only MC events
  if(!fPi0Only)
  {
    AnalyseBeamPFP(*beamParticle, evt);
  }
  else
  {
    // we want to process the beam like a daughter event for pi0 only particle gun tests
    AnalyseDaughterPFP(*beamParticle, evt, detProp, hitResults);
    beamStartPosX = -999;
    beamStartPosY = -999;
    beamStartPosZ = -999;
    beamEndPosX = -999;
    beamEndPosY = -999;
    beamEndPosZ = -999;
  }

  std::cout << beamParticle->Daughters().size() << std::endl;

  // analyse each daughter PFParticle from the beam
  for( size_t daughterID : beamParticle->Daughters() )
  {
    const recob::PFParticle * daughterPFP = &(pfpVec.at( daughterID ));
    AnalyseDaughterPFP(*daughterPFP, evt, detProp, hitResults);
  }

  // store any MC reated information here i.e. MC truth info
  if(!evt.isRealData())
  {
    if(!fPi0Only)
    {
      AnalyseMCTruthBeam(evt);
    }
    else
    {
      AnalyseMCTruth(*beamParticle, evt, clockData);
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

    // backtrack each daughter PFParticle from the beam
    for( size_t daughterID : beamParticle->Daughters() )
    {
      const recob::PFParticle * daughterPFP = &(pfpVec.at( daughterID ));
      AnalyseMCTruth(*daughterPFP, evt, clockData);
    }
  }
}

void protoana::pi0TestSelection::beginJob()
{
  // intiialize output root file
  art::ServiceHandle<art::TFileService> tfs;
  fOutTree = tfs->make<TTree>("beamana", "");
  //Once we are done, write results into the ROOT file
  fOutTree->Branch("Run", &run);
  fOutTree->Branch("SubRun", &subRun);
  fOutTree->Branch("EventID", &eventID);
  fOutTree->Branch("totalEvents", &totalEvents);
  fOutTree->Branch("beamEvents", &beamEvents);
  fOutTree->Branch("beamNum", &beam);
  fOutTree->Branch("pdgCode", &pdgCodes);

  // track-shower identification
  fOutTree->Branch("pandoraTag", &pandoraTags);
  fOutTree->Branch("reco_daughter_PFP_emScore_collection", &emScore);
  fOutTree->Branch("reco_daughter_PFP_trackScore_collection", &trackScore);
  fOutTree->Branch("CNNScore_collection", &CNNScore);

  // shower start position
  fOutTree->Branch("reco_daughter_allShower_startX", &startPosX);
  fOutTree->Branch("reco_daughter_allShower_startY", &startPosY);
  fOutTree->Branch("reco_daughter_allShower_startZ", &startPosZ);
  
  // shower direction
  fOutTree->Branch("reco_daughter_allShower_dirX", &dirX);
  fOutTree->Branch("reco_daughter_allShower_dirY", &dirY);
  fOutTree->Branch("reco_daughter_allShower_dirZ", &dirZ);
  
  // cone angle
  fOutTree->Branch("reco_daughter_allShower_coneAngle", &coneAngle);
  // length
  fOutTree->Branch("reco_daughter_allShower_length", &length);

  // hit/energy quantities
  fOutTree->Branch("reco_daughter_PFP_nHits_collection", &nHits);
  fOutTree->Branch("reco_daughter_allShower_energy", &energy);

  // quantity used to calculate the number of start hits
  fOutTree->Branch("hitRadial", &hitRadial);
  fOutTree->Branch("hitLongitudinal", &hitLongitudinal);
  fOutTree->Branch("reco_daughter_allShower_spacePointX", &spacePointX);
  fOutTree->Branch("reco_daughter_allShower_spacePointY", &spacePointY);
  fOutTree->Branch("reco_daughter_allShower_spacePointZ", &spacePointZ);

  // beam start position
  fOutTree->Branch("reco_beam_startX", &beamStartPosX);
  fOutTree->Branch("reco_beam_startY", &beamStartPosY);
  fOutTree->Branch("reco_beam_startZ", &beamStartPosZ);

  // beam end position
  fOutTree->Branch("reco_beam_endX", &beamEndPosX);
  fOutTree->Branch("reco_beam_endY", &beamEndPosY);
  fOutTree->Branch("reco_beam_endZ", &beamEndPosZ);

  // true start position
  fOutTree->Branch("reco_daughter_PFP_true_byHits_startX", &trueDaughterStartPosX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_startY", &trueDaughterStartPosY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_startZ", &trueDaughterStartPosZ);

  // true end position
  fOutTree->Branch("reco_daughter_PFP_true_byHits_endX", &trueDaughterEndPosX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_endY", &trueDaughterEndPosY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_endZ", &trueDaughterEndPosZ);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_startE", &trueDaughterEnergy);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_pdg", &trueDaughterPdg);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_mass", &trueDaughterMass);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_pX", &trueDaughterMomentumX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_pY", &trueDaughterMomentumY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_pZ", &trueDaughterMomentumZ);

  // true start position
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_startX", &trueParentStartPosX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_startY", &trueParentStartPosY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_startZ", &trueParentStartPosZ);

  // true end position
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_endX", &trueParentEndPosX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_endY", &trueParentEndPosY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_endZ", &trueParentEndPosZ);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_startE", &trueParentEnergy);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_pdg", &trueParentPdg);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_mass", &trueParentMass);

  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_pX", &trueParentMomentumX);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_pY", &trueParentMomentumY);
  fOutTree->Branch("reco_daughter_PFP_true_byHits_parent_pZ", &trueParentMomentumZ);

  // true start position
  fOutTree->Branch("reco_beam_PFP_true_byHits_startX", &trueBeamStartPosX);
  fOutTree->Branch("reco_beam_PFP_true_byHits_startY", &trueBeamStartPosY);
  fOutTree->Branch("reco_beam_PFP_true_byHits_startZ", &trueBeamStartPosZ);

  // true end position
  fOutTree->Branch("reco_beam_PFP_true_byHits_endX", &trueBeamEndPosX);
  fOutTree->Branch("reco_beam_PFP_true_byHits_endY", &trueBeamEndPosY);
  fOutTree->Branch("reco_beam_PFP_true_byHits_endZ", &trueBeamEndPosZ);

  fOutTree->Branch("reco_beam_PFP_true_byHits_startE", &trueBeamEnergy);
  fOutTree->Branch("reco_beam_PFP_true_byHits_pdg", &trueBeamPdg);
  fOutTree->Branch("reco_beam_PFP_true_byHits_mass", &trueBeamMass);


  fOutTree->Branch("g4_startX", &G4ParticleStartPosX);
  fOutTree->Branch("g4_startY", &G4ParticleStartPosY);
  fOutTree->Branch("g4_startZ", &G4ParticleStartPosZ);

  fOutTree->Branch("g4_endX", &G4ParticleEndPosX);
  fOutTree->Branch("g4_endY", &G4ParticleEndPosY);
  fOutTree->Branch("g4_endZ", &G4ParticleEndPosZ);

  fOutTree->Branch("g4_pX", &G4ParticleMomX);
  fOutTree->Branch("g4_pY", &G4ParticleMomY);
  fOutTree->Branch("g4_pZ", &G4ParticleMomZ);

  fOutTree->Branch("g4_Pdg", &G4ParticlePdg);
  fOutTree->Branch("g4_startE", &G4ParticleEnergy);
  fOutTree->Branch("g4_mass", &G4ParticleMass);
  fOutTree->Branch("g4_num", &G4ParticleNum);
  fOutTree->Branch("g4_mother", &G4ParticleMother);

  fOutTree->Branch("g4_matched_daughter_pdg", &matchedG4DaughterPdg);
  fOutTree->Branch("g4_matched_parent_pdg", &matchedG4ParentPdg);

  fOutTree->Branch("reco_PFP_ID", &PFPNum);
  fOutTree->Branch("reco_PFP_Mother", &PFPMother);

}


void protoana::pi0TestSelection::analyze(art::Event const & evt)
{
  //-----------------------------------------------//
  std::cout << "module running..." << std::endl;
  reset(); // clear any outputs that are lists
  
  // print metadata
  run = evt.run();
  std::cout << "run: " << run << std::endl;
  subRun = evt.subRun();
  std::cout << "subrun: " << subRun << std::endl;
  eventID = evt.id().event();
  std::cout << "event: " << eventID << std::endl;
  //-----------------------------------------------//

  //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
  // get various information needed to retrieve necessary data
  std::cout << "getting handle" << std::endl;
  art::ValidHandle<std::vector<recob::PFParticle> > pfpVec = evt.getValidHandle<std::vector<recob::PFParticle> >( fPFParticleTag ); // object to allow us to reference the PFParticles in the event
  std::cout << "number of PFParticles: " << pfpVec->size() << std::endl;
  std::cout << "got handle" << std::endl;

  std::cout << "getting clockData" << std::endl;
  auto const clockData = art::ServiceHandle<detinfo::DetectorClocksService>()->DataFor(evt); // use timing to match PFP to MC
  std::cout << "got clockData" << std::endl;
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
  for(recob::PFParticle pfp : *pfpVec)
  {
    const int self = pfp.Self();
    int parent = pfp.Parent();
    // print some information for debugging
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "PFP number: " << self << std::endl;
    std::cout << "is primary? " << pfp.IsPrimary() << std::endl;
    std::cout << "Number of daughters: " << pfp.NumDaughters() << std::endl;
    if(self == beam)
    {
      std::cout << "beam particle: " << self << std::endl;
    }

    // make it clear that the particle has no parent
    if(pfp.Parent() > pfpVec->size())
    {
      parent = -999;
    }
    
    std::cout << "parent: " << parent << std::endl;

    PFPNum.push_back(self);
    PFPMother.push_back(parent);

    AnalyseDaughterPFP(pfp, evt, detProp, hitResults);
    if(!evt.isRealData())
    {
      AnalyseMCTruth(pfp, evt, clockData);
    }
    std::cout << "----------------------------------------" << std::endl;
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

  totalEvents++;
  fOutTree->Fill(); // fill the root tree with the outputs
}

// Maybe do some stuff here???
void protoana::pi0TestSelection::endJob()
{

}

DEFINE_ART_MODULE(protoana::pi0TestSelection)
