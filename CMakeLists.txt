art_make(Pi0Analyser "module"
  BASENAME_ONLY
  MODULE_LIBRARIES
  ${ART_FRAMEWORK_CORE}
  ${ART_FRAMEWORK_PRINCIPAL}
  ${ART_FRAMEWORK_SERVICES_REGISTRY}
  ${ART_FRAMEWORK_SERVICES_BASIC}
  ${ART_ROOT_IO_TFILE_SUPPORT} ${ROOT_CORE}
  ${ART_ROOT_IO_TFILESERVICE_SERVICE}
  ${ART_PERSISTENCY_COMMON}
  canvas
  art_Persistency_Provenance
  ${ART_UTILITIES}
  ${MF_MESSAGELOGGER}
  cetlib cetlib_except
  lardataobj_RecoBase
  lardata_Utilities
  larcorealg_Geometry
  nusimdata_SimulationBase
  larsim_MCCheater_PhotonBackTrackerService_service           
  larsim_MCCheater_BackTrackerService_service           
  larsim_MCCheater_ParticleInventoryService_service
  larreco_RecoAlg
  ${ROOT_BASIC_LIB_LIST}
  #ProtoDUNEAnaUtils
  ProtoDUNEDataUtils
  ProtoDUNEUtilities
  )
install_headers()
install_fhicl()
install_source()

