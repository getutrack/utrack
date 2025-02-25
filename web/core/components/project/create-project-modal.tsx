import { useEffect, FC, useState } from "react";
// utrack ui
import { EModalPosition, EModalWidth, ModalCore } from "@utrack/ui";
// helpers
import { getAssetIdFromUrl } from "@/helpers/file.helper";
import { checkURLValidity } from "@/helpers/string.helper";
// utrack web components
import { CreateProjectForm } from "@/utrack-web/components/projects/create/root";
// utrack web types
import { TProject } from "@/utrack-web/types/projects";
// services
import { FileService } from "@/services/file.service";
const fileService = new FileService();
import { ProjectFeatureUpdate } from "./project-feature-update";

type Props = {
  isOpen: boolean;
  onClose: () => void;
  setToFavorite?: boolean;
  workspaceSlug: string;
  data?: Partial<TProject>;
};

enum EProjectCreationSteps {
  CREATE_PROJECT = "CREATE_PROJECT",
  FEATURE_SELECTION = "FEATURE_SELECTION",
}

export const CreateProjectModal: FC<Props> = (props) => {
  const { isOpen, onClose, setToFavorite = false, workspaceSlug, data } = props;
  // states
  const [currentStep, setCurrentStep] = useState<EProjectCreationSteps>(EProjectCreationSteps.CREATE_PROJECT);
  const [createdProjectId, setCreatedProjectId] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setCurrentStep(EProjectCreationSteps.CREATE_PROJECT);
      setCreatedProjectId(null);
    }
  }, [isOpen]);

  const handleNextStep = (projectId: string) => {
    if (!projectId) return;
    setCreatedProjectId(projectId);
    setCurrentStep(EProjectCreationSteps.FEATURE_SELECTION);
  };

  const handleCoverImageStatusUpdate = async (projectId: string, coverImage: string) => {
    if (!checkURLValidity(coverImage)) {
      await fileService.updateBulkProjectAssetsUploadStatus(workspaceSlug, projectId, projectId, {
        asset_ids: [getAssetIdFromUrl(coverImage)],
      });
    }
  };

  return (
    <ModalCore isOpen={isOpen} handleClose={onClose} position={EModalPosition.TOP} width={EModalWidth.XXL}>
      {currentStep === EProjectCreationSteps.CREATE_PROJECT && (
        <CreateProjectForm
          setToFavorite={setToFavorite}
          workspaceSlug={workspaceSlug}
          onClose={onClose}
          updateCoverImageStatus={handleCoverImageStatusUpdate}
          handleNextStep={handleNextStep}
          data={data}
        />
      )}
      {currentStep === EProjectCreationSteps.FEATURE_SELECTION && (
        <ProjectFeatureUpdate projectId={createdProjectId} workspaceSlug={workspaceSlug} onClose={onClose} />
      )}
    </ModalCore>
  );
};
