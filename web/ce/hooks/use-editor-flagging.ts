// editor
import { TExtensions } from "@youtrack/editor";

/**
 * @description extensions disabled in various editors
 */
export const useEditorFlagging = (): {
  documentEditor: TExtensions[];
  richTextEditor: TExtensions[];
} => ({
  documentEditor: ["ai", "collaboration-cursor"],
  richTextEditor: ["ai", "collaboration-cursor"],
});
