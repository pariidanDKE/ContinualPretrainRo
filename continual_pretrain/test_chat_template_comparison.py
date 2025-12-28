"""
Test to compare native instruction model chat templates with our custom DataPreprocessor templates.

This ensures that our custom templates (used for PT models) match the native templates
(used for IT models) so that both model types are evaluated in the same format.
"""

from transformers import AutoTokenizer
from data_module import DataPreprocessor

def test_template_comparison(model_name, test_messages):
    """Compare native chat template with our custom DataPreprocessor template"""

    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=None)

    # Get native chat template output (if exists)
    has_native_template = tokenizer.chat_template is not None

    if has_native_template:
        print("\n✅ Model has native chat template")
        native_output = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        print("\n❌ Model has NO native chat template")
        native_output = None

    # Get our custom template output
    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        text_field="formatted_text",
        add_bos_eos=False,
        use_sft_config=False
    )

    if preprocessor.special_map is not None:
        custom_template = preprocessor.get_chat_template()

        # Temporarily set our custom template
        original_template = tokenizer.chat_template
        tokenizer.chat_template = custom_template

        custom_output = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Restore original template
        tokenizer.chat_template = original_template

        print(f"✅ Generated custom template for {preprocessor.family} family")
    else:
        print("❌ Could not detect model family")
        custom_output = None

    # Display outputs
    print("\n" + "-"*80)
    print("NATIVE TEMPLATE OUTPUT:")
    print("-"*80)
    if native_output:
        print(repr(native_output))
        print("\nFormatted view:")
        print(native_output)
    else:
        print("(No native template)")

    print("\n" + "-"*80)
    print("CUSTOM TEMPLATE OUTPUT:")
    print("-"*80)
    if custom_output:
        print(repr(custom_output))
        print("\nFormatted view:")
        print(custom_output)
    else:
        print("(No custom template)")

    # Compare
    print("\n" + "="*80)
    if native_output and custom_output:
        if native_output == custom_output:
            print("✅ MATCH: Templates produce identical output!")
            return True
        else:
            print("❌ MISMATCH: Templates produce different output!")

            # Show character-by-character diff
            print("\nCharacter-by-character comparison:")
            max_len = max(len(native_output), len(custom_output))

            diff_found = False
            for i in range(max_len):
                native_char = native_output[i] if i < len(native_output) else "EOF"
                custom_char = custom_output[i] if i < len(custom_output) else "EOF"

                if native_char != custom_char:
                    if not diff_found:
                        print(f"\nFirst difference at position {i}:")
                        diff_found = True

                    if i < 10 or not diff_found:  # Show first 10 differences
                        print(f"  Position {i}: Native={repr(native_char)} vs Custom={repr(custom_char)}")

                    if i >= 10 and diff_found:
                        print(f"  ... (showing only first differences)")
                        break

            # Show length difference
            print(f"\nLength: Native={len(native_output)}, Custom={len(custom_output)}")

            return False
    elif not native_output and custom_output:
        print("⚠️  Native template missing, but custom template works!")
        return True
    elif native_output and not custom_output:
        print("❌ Custom template generation failed!")
        return False
    else:
        print("❌ Both templates failed!")
        return False


if __name__ == "__main__":
    # Test messages in Romanian (matching your training data format)
    test_messages = [
        {"role": "system", "content": "Ești un asistent util care răspunde în limba română."},
        {"role": "user", "content": "Salut! Cum te cheamă?"},
        {"role": "assistant", "content": "Bună ziua! Mă numesc Claude. Cu ce te pot ajuta astăzi?"},
        {"role": "user", "content": "Care este capitala României?"},
        {"role": "assistant", "content": "Capitala României este București."}
    ]

    # Test all three model families
    models_to_test = [
        ("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 Instruct"),
        ("google/gemma-3-1b-it", "Gemma 3 IT"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen 2.5 Instruct"),
    ]

    results = {}

    for model_path, display_name in models_to_test:
        try:
            match = test_template_comparison(model_path, test_messages)
            results[display_name] = match
        except Exception as e:
            print(f"\n❌ Error testing {display_name}: {e}")
            import traceback
            traceback.print_exc()
            results[display_name] = False

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for model_name, matched in results.items():
        status = "✅ PASS" if matched else "❌ FAIL"
        print(f"{status}: {model_name}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your custom templates match the native instruction templates.")
        print("PT and IT models will be evaluated in the same format!")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("Review the differences above to fix the custom templates.")
    print("="*80)
