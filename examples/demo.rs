use minilamb::{parse, parse_and_evaluate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== minilamb Parser Demo ===\n");

    // Test different lambda symbols
    println!("Testing alternative lambda symbols:");
    let expressions = ["λx.x", "\\x.x", "/x.x", "|x.x"];
    for expr in &expressions {
        let parsed = parse(expr)?;
        println!("  {expr} -> {parsed}");
    }

    // Test De Bruijn vs Named variables
    println!("\nTesting De Bruijn vs Named variables:");
    let de_bruijn = parse("λ.1")?;
    let named = parse("λx.x")?;
    println!("  De Bruijn: λ.1 -> {de_bruijn}");
    println!("  Named:     λx.x -> {named}");
    let equal = de_bruijn == named;
    println!("  Equal? {equal}");

    // Test complex expressions
    println!("\nTesting complex expressions:");

    // Church boolean TRUE
    let church_true = parse("λt.λf.t")?;
    println!("  Church TRUE: λt.λf.t -> {church_true}");

    // Church boolean FALSE
    let church_false = parse("λt.λf.f")?;
    println!("  Church FALSE: λt.λf.f -> {church_false}");

    // Church numeral 2
    let church_two = parse("λf.λx.f (f x)")?;
    println!("  Church 2: λf.λx.f (f x) -> {church_two}");

    // Test evaluation
    println!("\nTesting evaluation:");

    // Identity function applied to itself: (λx.x) (λy.y) -> (λy.y)
    let identity_expr = "(λx.x) (λy.y)";
    let result = parse_and_evaluate(identity_expr, 100)?;
    println!("  {identity_expr} -> {result}");

    // Application associativity: f g h -> ((f g) h)
    let application = parse("λx.λy.λz.x y z")?;
    println!("  Application: λx.λy.λz.x y z -> {application}");

    println!("\n=== All tests passed! ===");
    Ok(())
}
