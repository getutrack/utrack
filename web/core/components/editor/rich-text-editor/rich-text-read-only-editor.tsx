import React from "react";
// editor
import { EditorReadOnlyRefApi, IRichTextReadOnlyEditor, RichTextReadOnlyEditorWithRef } from "@utrack/editor";
// helpers
import { cn } from "@/helpers/common.helper";
import { getReadOnlyEditorFileHandlers } from "@/helpers/editor.helper";
// hooks
import { useMention } from "@/hooks/store";

type RichTextReadOnlyEditorWrapperProps = Omit<IRichTextReadOnlyEditor, "fileHandler" | "mentionHandler"> & {
  workspaceSlug: string;
  projectId?: string;
};

export const RichTextReadOnlyEditor = React.forwardRef<EditorReadOnlyRefApi, RichTextReadOnlyEditorWrapperProps>(
  ({ workspaceSlug, projectId, ...props }, ref) => {
    const { mentionHighlights } = useMention({});

    return (
      <RichTextReadOnlyEditorWithRef
        ref={ref}
        fileHandler={getReadOnlyEditorFileHandlers({
          projectId,
          workspaceSlug,
        })}
        mentionHandler={{
          highlights: mentionHighlights,
        }}
        {...props}
        // overriding the containerClassName to add relative class passed
        containerClassName={cn(props.containerClassName, "relative pl-3")}
      />
    );
  }
);

RichTextReadOnlyEditor.displayName = "RichTextReadOnlyEditor";
